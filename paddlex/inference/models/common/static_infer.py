# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Union, Sequence, Tuple, List
from pathlib import Path

import paddle
import numpy as np
from paddle.inference import Config, create_predictor
from typing_extensions import assert_never

from ....utils import logging
from ...utils.pp_option import PaddlePredictorOption
from ...utils.hpi import (
    MBIConfig,
    ONNXRuntimeConfig,
    OpenVINOConfig,
    TensorRTConfig,
    OMConfig,
    get_model_paths,
    suggest_inference_backend_and_config,
)


CACHE_DIR = ".cache"


# XXX: Better use Paddle Inference API to do this
def _pd_dtype_to_np_dtype(pd_dtype):
    if pd_dtype == paddle.inference.DataType.FLOAT64:
        return np.float64
    elif pd_dtype == paddle.inference.DataType.FLOAT32:
        return np.float32
    elif pd_dtype == paddle.inference.DataType.INT64:
        return np.int64
    elif pd_dtype == paddle.inference.DataType.INT32:
        return np.int32
    elif pd_dtype == paddle.inference.DataType.UINT8:
        return np.uint8
    elif pd_dtype == paddle.inference.DataType.INT8:
        return np.int8
    else:
        raise TypeError(f"Unsupported data type: {pd_dtype}")


def _collect_trt_shape_range_info(
    model_file,
    model_params,
    gpu_id,
    shape_range_info_path,
    dynamic_shapes,
    dynamic_shape_input_data,
):
    dynamic_shape_input_data = dynamic_shape_input_data or {}

    config = paddle.inference.Config(model_file, model_params)
    config.enable_use_gpu(100, gpu_id)
    config.collect_shape_range_info(shape_range_info_path)
    # TODO: Add other needed options
    config.disable_glog_info()
    predictor = paddle.inference.create_predictor(config)

    input_names = predictor.get_input_names()
    for name in dynamic_shapes:
        if name not in input_names:
            raise ValueError(
                f"Invalid input name {repr(name)} found in `dynamic_shapes`"
            )
    for name in input_names:
        if name not in dynamic_shapes:
            raise ValueError(f"Input name {repr(name)} not found in `dynamic_shapes`")
    for name in dynamic_shape_input_data:
        if name not in input_names:
            raise ValueError(
                f"Invalid input name {repr(name)} found in `dynamic_shape_input_data`"
            )
    # It would be better to check if the shapes are valid.

    min_arrs, opt_arrs, max_arrs = {}, {}, {}
    for name, candidate_shapes in dynamic_shapes.items():
        # XXX: Currently we have no way to get the data type of the tensor
        # without creating an input handle.
        handle = predictor.get_input_handle(name)
        dtype = _pd_dtype_to_np_dtype(handle.type())
        min_shape, opt_shape, max_shape = candidate_shapes
        if name in dynamic_shape_input_data:
            min_arrs[name] = np.array(
                dynamic_shape_input_data[name][0], dtype=dtype
            ).reshape(min_shape)
            opt_arrs[name] = np.array(
                dynamic_shape_input_data[name][1], dtype=dtype
            ).reshape(opt_shape)
            max_arrs[name] = np.array(
                dynamic_shape_input_data[name][2], dtype=dtype
            ).reshape(max_shape)
        else:
            min_arrs[name] = np.ones(min_shape, dtype=dtype)
            opt_arrs[name] = np.ones(opt_shape, dtype=dtype)
            max_arrs[name] = np.ones(max_shape, dtype=dtype)

    # `opt_arrs` is used twice to ensure it is the most frequently used.
    for arrs in [min_arrs, opt_arrs, opt_arrs, max_arrs]:
        for name, arr in arrs.items():
            handle = predictor.get_input_handle(name)
            handle.reshape(arr.shape)
            handle.copy_from_cpu(arr)
        predictor.run()

    # HACK: The shape range info will be written to the file only when
    # `predictor` is garbage collected. It works in CPython, but it is
    # definitely a bad idea to count on the implementation-dependent behavior of
    # a garbage collector. Is there a more explicit and deterministic way to
    # handle this?


class PaddleCopy2GPU:
    def __init__(self, input_handlers):
        super().__init__()
        self.input_handlers = input_handlers

    def __call__(self, x):
        for idx in range(len(x)):
            self.input_handlers[idx].reshape(x[idx].shape)
            self.input_handlers[idx].copy_from_cpu(x[idx])


class PaddleCopy2CPU:
    def __init__(self, output_handlers):
        super().__init__()
        self.output_handlers = output_handlers

    def __call__(self):
        output = []
        for out_tensor in self.output_handlers:
            batch = out_tensor.copy_to_cpu()
            output.append(batch)
        return output


class PaddleModelInfer:
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def __call__(self):
        self.predictor.run()


class StaticInfer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, x: Sequence[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError


class PaddleInfer(StaticInfer):
    def __init__(
        self,
        model_dir: str,
        model_file_prefix: str,
        option: PaddlePredictorOption,
    ) -> None:
        super().__init__()
        self.model_dir = model_dir
        self.model_file_prefix = model_file_prefix
        self._update_option(option)

    @property
    def option(self) -> PaddlePredictorOption:
        return self._option if hasattr(self, "_option") else None

    # TODO: We should re-evaluate whether allowing changes to `option` across
    # different calls provides any benefits.
    @option.setter
    def option(self, option: Union[None, PaddlePredictorOption]) -> None:
        if option:
            self._update_option(option)

    @property
    def benchmark(self):
        return {
            "Copy2GPU": self.copy2gpu,
            "Infer": self.infer,
            "Copy2CPU": self.copy2cpu,
        }

    def __call__(self, x: Sequence[np.ndarray]) -> List[np.ndarray]:
        self.copy2gpu(x)
        self.infer()
        pred = self.copy2cpu()
        return pred

    def _update_option(self, option: PaddlePredictorOption) -> None:
        if self.option and option == self.option:
            return
        self._option = option
        self._reset()

    def _reset(self) -> None:
        logging.debug(f"Env: {self.option}")
        (
            predictor,
            input_handlers,
            output_handlers,
        ) = self._create()
        # TODO: Would a more generic name like `copy2device` be better here?
        self.copy2gpu = PaddleCopy2GPU(input_handlers)
        self.copy2cpu = PaddleCopy2CPU(output_handlers)
        self.infer = PaddleModelInfer(predictor)

    def _create(
        self,
    ) -> Tuple[
        "paddle.base.libpaddle.PaddleInferPredictor",
        "paddle.base.libpaddle.PaddleInferTensor",
        "paddle.base.libpaddle.PaddleInferTensor",
    ]:
        """_create"""
        model_paths = get_model_paths(self.model_dir, self.model_file_prefix)
        if "paddle" not in model_paths:
            raise RuntimeError("No valid Paddle model found")
        model_file, params_file = model_paths["paddle"]

        config = Config(str(model_file), str(params_file))

        config.enable_memory_optim()
        if self.option.device_type in ("gpu", "dcu"):
            if self.option.device_type == "gpu":
                config.exp_disable_mixed_precision_ops({"feed", "fetch"})
            config.enable_use_gpu(100, self.option.device_id or 0)
            if self.option.device_type == "gpu":
                precision_map = {
                    "trt_int8": Config.Precision.Int8,
                    "trt_fp32": Config.Precision.Float32,
                    "trt_fp16": Config.Precision.Half,
                }
                if self.option.run_mode in precision_map.keys():
                    cache_dir = self.model_dir / CACHE_DIR / "paddle_inference"
                    self._configure_trt(
                        config,
                        precision_map[self.option.run_mode],
                        model_file,
                        params_file,
                        cache_dir,
                    )

        elif self.option.device_type == "npu":
            config.enable_custom_device("npu")
        elif self.option.device_type == "xpu":
            pass
        elif self.option.device_type == "mlu":
            config.enable_custom_device("mlu")
        else:
            assert self.option.device_type == "cpu"
            config.disable_gpu()
            if "mkldnn" in self.option.run_mode:
                try:
                    config.enable_mkldnn()
                    if "bf16" in self.option.run_mode:
                        config.enable_mkldnn_bfloat16()
                except Exception as e:
                    logging.warning(
                        "MKL-DNN is not available. We will disable MKL-DNN."
                    )
                config.set_mkldnn_cache_capacity(-1)
            else:
                if hasattr(config, "disable_mkldnn"):
                    config.disable_mkldnn()

        if self.option.disable_glog_info:
            config.disable_glog_info()

        config.set_cpu_math_library_num_threads(self.option.cpu_threads)

        if self.option.device_type in ("cpu", "gpu"):
            if not (
                self.option.device_type == "gpu"
                and self.option.run_mode.startswith("trt")
            ):
                if hasattr(config, "enable_new_ir"):
                    config.enable_new_ir(self.option.enable_new_ir)
                if hasattr(config, "enable_new_executor"):
                    config.enable_new_executor()
                config.set_optimization_level(3)

        for del_p in self.option.delete_pass:
            config.delete_pass(del_p)

        if self.option.device_type in ("gpu", "dcu"):
            if paddle.is_compiled_with_rocm():
                # Delete unsupported passes in dcu
                config.delete_pass("conv2d_add_act_fuse_pass")
                config.delete_pass("conv2d_add_fuse_pass")

        predictor = create_predictor(config)

        # Get input and output handlers
        input_names = predictor.get_input_names()
        input_names.sort()
        input_handlers = []
        output_handlers = []
        for input_name in input_names:
            input_handler = predictor.get_input_handle(input_name)
            input_handlers.append(input_handler)
        output_names = predictor.get_output_names()
        for output_name in output_names:
            output_handler = predictor.get_output_handle(output_name)
            output_handlers.append(output_handler)
        return predictor, input_handlers, output_handlers

    def _configure_trt(
        self, config, precision_mode, model_file, params_file, cache_dir
    ):
        config.set_optim_cache_dir(str(cache_dir / "optim_cache"))

        config.enable_tensorrt_engine(
            workspace_size=self.option.trt_max_workspace_size,
            max_batch_size=self.option.trt_max_batch_size,
            min_subgraph_size=self.option.trt_min_subgraph_size,
            precision_mode=precision_mode,
            use_static=self.option.trt_use_static,
            use_calib_mode=self.option.trt_use_calib_mode,
        )

        if self.option.trt_use_dynamic_shapes:
            if self.option.trt_collect_shape_range_info:
                # NOTE: We always use a shape range info file.
                if self.option.trt_shape_range_info_path is not None:
                    trt_shape_range_info_path = Path(
                        self.option.trt_shape_range_info_path
                    )
                else:
                    trt_shape_range_info_path = cache_dir / "shape_range_info.pbtxt"
                should_collect_shape_range_info = True
                if not trt_shape_range_info_path.exists():
                    trt_shape_range_info_path.parent.mkdir(parents=True, exist_ok=True)
                    logging.info(
                        f"Shape range info will be collected into {trt_shape_range_info_path}"
                    )
                elif self.option.trt_discard_cached_shape_range_info:
                    trt_shape_range_info_path.unlink()
                    logging.info(
                        f"The shape range info file ({trt_shape_range_info_path}) has been removed, and the shape range info will be re-collected."
                    )
                else:
                    logging.info(
                        f"A shape range info file ({trt_shape_range_info_path}) already exists. There is no need to collect the info again."
                    )
                    should_collect_shape_range_info = False
                if should_collect_shape_range_info:
                    _collect_trt_shape_range_info(
                        str(model_file),
                        str(params_file),
                        self.option.device_id or 0,
                        str(trt_shape_range_info_path),
                        self.option.trt_dynamic_shapes,
                        self.option.trt_dynamic_shape_input_data,
                    )
                config.enable_tuned_tensorrt_dynamic_shape(
                    str(trt_shape_range_info_path),
                    self.option.trt_allow_rebuild_at_runtime,
                )
            else:
                if self.option.trt_dynamic_shapes is not None:
                    min_shapes, opt_shapes, max_shapes = {}, {}, {}
                    for (
                        key,
                        shapes,
                    ) in self.option.trt_dynamic_shapes.items():
                        min_shapes[key] = shapes[0]
                        opt_shapes[key] = shapes[1]
                        max_shapes[key] = shapes[2]
                        config.set_trt_dynamic_shape_info(
                            min_shapes, max_shapes, opt_shapes
                        )
                else:
                    raise RuntimeError("No dynamic shape information provided")


class MultibackendInfer(StaticInfer):
    def __init__(
        self,
        model_dir: str,
        model_file_prefix: str,
        config: MBIConfig,
    ) -> None:
        super().__init__()
        self._model_dir = model_dir
        self._model_file_prefix = model_file_prefix
        self._config = config
        self._ui_option = self._prepare_ui_option()
        self._ui_runtime = self._build_ui_runtime(self._ui_option)

    @property
    def model_dir(self) -> str:
        return self._model_dir

    @property
    def model_file_prefix(self) -> str:
        return self._model_file_prefix

    @property
    def config(self) -> MBIConfig:
        return self._config

    def __call__(self, x: Sequence[np.ndarray]) -> List[np.ndarray]:
        num_inputs = self._ui_runtime.num_inputs()
        if len(x) != num_inputs:
            raise ValueError(f"Expected {num_inputs} inputs but got {len(x)} instead")
        inputs = {}
        for idx, input_ in enumerate(x):
            input_name = self._ui_runtime.get_input_info(idx).name
            input_ = np.ascontiguousarray(input_)
            inputs[input_name] = input_
        outputs = self._ui_runtime.infer(inputs)
        return outputs

    def _prepare_ui_option(self, ui_option=None):
        from ultra_infer import (
            ModelFormat,
            RuntimeOption,
            is_built_with_om,
            is_built_with_openvino,
            is_built_with_ort,
            is_built_with_trt,
        )

        model_paths = get_model_paths(self.model_dir, self.model_file_prefix)
        available_backends = []
        if is_built_with_openvino() and "onnx" in model_paths:
            available_backends.append("openvino")
        if is_built_with_ort() and "onnx" in model_paths:
            available_backends.append("onnxruntime")
        if is_built_with_trt() and "onnx" in model_paths:
            available_backends.append("tensorrt")
        if is_built_with_om() and "om" in model_paths:
            available_backends.append("om")

        if not available_backends:
            raise RuntimeError("No inference backend is available")

        if self._config.backend not in available_backends:
            raise RuntimeError(
                f"Unavailable inference backend {repr(self._config.backend)}"
            )

        if self._config.auto_config:
            # Should we use the strategy pattern here to allow extensible
            # strategies?
            ret = suggest_inference_backend_and_config(
                self._config, available_backends=available_backends
            )
            if ret[0] is None:
                # Should I use a custom exception?
                raise RuntimeError(
                    f"No inference backend and configuration could be suggested. Reason: {ret[1]}"
                )
            backend, backend_config = ret
        else:
            backend = self._config.backend
            if backend is None:
                raise RuntimeError(
                    "When automatic configuration is not used, the inference backend must be specified manually."
                )
            backend_config = self._config.backend_config or {}

        if ui_option is None:
            ui_option = RuntimeOption()

        if self._config.device_type == "cpu":
            pass
        elif self._config.device_type == "gpu":
            ui_option.use_gpu(self._config.device_id or 0)
        elif self._config.device_type == "npu":
            ui_option.use_ascend()
        else:
            raise RuntimeError(
                f"Unsupported device type {repr(self._config.device_type)}"
            )

        if backend in ("openvino", "onnxruntime", "tensorrt"):
            assert "onnx" in model_paths
            ui_option.set_model_path(str(model_paths["onnx"]), "", ModelFormat.ONNX)
        elif backend == "om":
            assert "om" in model_paths
            ui_option.set_model_path(str(model_paths["om"]), "", ModelFormat.OM)
        else:
            assert_never(backend)

        if backend == "openvino":
            backend_config = OpenVINOConfig.model_validate(backend_config)
            ui_option.use_openvino_backend()
            ui_option.set_cpu_thread_num(backend_config.cpu_num_threads)
        elif backend == "onnxruntime":
            backend_config = ONNXRuntimeConfig.model_validate(backend_config)
            ui_option.use_ort_backend()
            ui_option.set_cpu_thread_num(backend_config.cpu_num_threads)
        elif backend == "tensorrt":
            if (
                backend_config.get("use_dynamic_shapes", True)
                and backend_config.get("dynamic_shapes", None) is None
            ):
                trt_info = self._config.hpi_info.backend_configs.tensorrt
                if trt_info is not None and trt_info.dynamic_shapes is not None:
                    trt_dynamic_shapes = trt_info.dynamic_shapes
                    logging.debug(
                        "TensorRT dynamic shapes set to %s", trt_dynamic_shapes
                    )
                    backend_config["dynamic_shapes"] = trt_dynamic_shapes
            backend_config = TensorRTConfig.model_validate(backend_config)
            ui_option.use_trt_backend()
            cache_dir = self.model_dir / CACHE_DIR / "tensorrt"
            cache_dir.mkdir(parents=True, exist_ok=True)
            ui_option.trt_option.serialize_file = str(cache_dir / "trt_serialized.trt")
            if backend_config.precision == "FP16":
                ui_option.trt_option.enable_fp16 = True
            if not backend_config.use_dynamic_shapes:
                raise RuntimeError(
                    "TensorRT static shape inference is currently not supported"
                )
            if backend_config.dynamic_shapes is not None:
                for name, shapes in backend_config.dynamic_shapes.items():
                    ui_option.trt_option.set_shape(name, *shapes)
        elif backend == "om":
            backend_config = OMConfig.model_validate(backend_config)
            ui_option.use_om_backend()
        else:
            raise RuntimeError(f"Unsupported inference backend {repr(backend)}")

        logging.info("Inference backend: %s", backend)
        logging.info("Inference backend config: %s", backend_config)

        return ui_option

    def _build_ui_runtime(self, ui_option):
        from ultra_infer import Runtime

        return Runtime(ui_option)

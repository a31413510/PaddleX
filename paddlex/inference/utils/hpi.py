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

import importlib.resources
import json
import platform
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Final, List, Literal, Optional, Tuple, TypedDict, Union

from pydantic import BaseModel
from typing_extensions import TypeAlias

from ...utils.flags import FLAGS_json_format_model


class PaddleInferenceInfo(BaseModel):
    trt_dynamic_shapes: Optional[Dict[str, List[List[int]]]] = None
    trt_dynamic_shape_input_data: Optional[Dict[str, List[List[float]]]] = None


class TensorRTInfo(BaseModel):
    dynamic_shapes: Optional[Dict[str, List[List[int]]]] = None


class InferenceBackendInfoCollection(BaseModel):
    paddle_infer: Optional[PaddleInferenceInfo] = None
    tensorrt: Optional[TensorRTInfo] = None


# Does using `TypedDict` make things more convenient?
class HPIInfo(BaseModel):
    backend_configs: Optional[InferenceBackendInfoCollection] = None


# For multi-backend inference only
InferenceBackend: TypeAlias = Literal["openvino", "onnxruntime", "tensorrt", "omruntime"]


class OpenVINOConfig(BaseModel):
    cpu_num_threads: int = 8


class ONNXRuntimeConfig(BaseModel):
    cpu_num_threads: int = 8

class OMRuntimeConfig(BaseModel):
    pass
    

class TensorRTConfig(BaseModel):
    precision: Literal["fp32", "fp16"] = "fp32"
    use_dynamic_shapes: bool = True
    dynamic_shapes: Optional[Dict[str, List[List[int]]]] = None
    # TODO: Control caching behavior


class MBIConfig(BaseModel):
    model_name: str
    device_type: str
    device_id: Optional[int] = None
    auto_config: bool = True
    backend: Optional[InferenceBackend] = None
    backend_config: Optional[Dict[str, Any]] = None
    hpi_info: Optional[HPIInfo] = None
    # TODO: Add more validation logic here


class ModelInfo(BaseModel):
    name: str
    hpi_info: Optional[HPIInfo] = None


ModelFormat: TypeAlias = Literal["paddle", "onnx", "om"]


class ModelPaths(TypedDict, total=False):
    paddle: Tuple[Path, Path]
    onnx: Path
    om: Path


def get_model_paths(
    model_dir: Union[str, PathLike], model_file_prefix: str
) -> ModelPaths:
    model_dir = Path(model_dir)
    model_paths: ModelPaths = {}
    pd_model_path = None
    if FLAGS_json_format_model:
        if (model_dir / f"{model_file_prefix}.json").exists():
            pd_model_path = model_dir / f"{model_file_prefix}.json"
    else:
        if (model_dir / f"{model_file_prefix}.json").exists():
            pd_model_path = model_dir / f"{model_file_prefix}.json"
        elif (model_dir / f"{model_file_prefix}.pdmodel").exists():
            pd_model_path = model_dir / f"{model_file_prefix}.pdmodel"
    if pd_model_path and (model_dir / f"{model_file_prefix}.pdiparams").exists():
        model_paths["paddle"] = (
            pd_model_path,
            model_dir / f"{model_file_prefix}.pdiparams",
        )
    if (model_dir / f"{model_file_prefix}.onnx").exists():
        model_paths["onnx"] = model_dir / f"{model_file_prefix}.onnx"
    if (model_dir / f"{model_file_prefix}.om").exists():
        model_paths["om"] = model_dir / f"{model_file_prefix}.om"
    return model_paths


_PREFERRED_INFERENCE_BACKENDS: Final[Dict[str, List[InferenceBackend]]] = {
    "cpu_x64": ["openvino", "onnxruntime"],
    "gpu_cuda118_cudnn86": ["tensorrt", "onnxruntime"],
}


def suggest_inference_backend_and_config(
    mbi_config: MBIConfig,
    available_backends: Optional[List[InferenceBackend]] = None,
) -> Union[Tuple[InferenceBackend, Dict[str, Any]], Tuple[None, str]]:
    # TODO: The current strategy is naive. It would be better to consider
    # additional important factors, such as NVIDIA GPU compute capability and
    # device manufacturers. We should also allow users to provide hints.

    import lazy_paddle as paddle

    if available_backends is not None and not available_backends:
        return None, "No inference backends are available."

    paddle_version = paddle.__version__
    if paddle_version != "3.0.0-rc0":
        return None, f"{repr(paddle_version)} is not a supported Paddle version."

    if mbi_config.device_type == "cpu":
        uname = platform.uname()
        arch = uname.machine.lower()
        if arch == "x86_64":
            key = "cpu_x64"
        else:
            return None, f"{repr(arch)} is not a supported architecture."
    elif mbi_config.device_type == "gpu":
        # Currently only NVIDIA GPUs are supported.
        # FIXME: We should not rely on the PaddlePaddle library to detemine CUDA
        # and cuDNN versions.
        # Should we inject environment info from the outside?
        from lazy_paddle import version as paddle_version

        cuda_version = paddle_version.cuda()
        cuda_version = cuda_version.replace(".", "")
        cudnn_version = paddle_version.cudnn().rsplit(".", 1)[0]
        cudnn_version = cudnn_version.replace(".", "")
        key = f"gpu_cuda{cuda_version}_cudnn{cudnn_version}"
    else:
        return None, f"{repr(mbi_config.device_type)} is not a supported device type."

    with importlib.resources.open_text(
        __package__, "mbi_model_info_collection.json", encoding="utf-8"
    ) as f:
        mbi_model_info_collection = json.load(f)

    if key not in mbi_model_info_collection:
        return None, "No prior knowledge can be utilized."
    mbi_model_info_collection_for_env = mbi_model_info_collection[key]

    if mbi_config.model_name not in mbi_model_info_collection_for_env:
        return None, f"{repr(mbi_config.model_name)} is not a known model."
    supported_pseudo_backends = mbi_model_info_collection_for_env[mbi_config.model_name]

    supported_backends = []
    for pb in supported_pseudo_backends:
        if pb == "tensorrt_fp16":
            backend = "tensorrt"
        else:
            backend = pb
        supported_backends.append(backend)

    assert key in _PREFERRED_INFERENCE_BACKENDS
    preferred_backends = _PREFERRED_INFERENCE_BACKENDS[key]
    assert all(backend in preferred_backends for backend in supported_backends)
    candidate_backends = sorted(
        supported_backends, key=lambda b: preferred_backends.index(b)
    )

    if available_backends is not None:
        candidate_backends = [
            backend for backend in candidate_backends if backend in available_backends
        ]

    if not candidate_backends:
        return None, "No inference backend can be selected."

    if mbi_config.backend is not None:
        if mbi_config.backend not in candidate_backends:
            return (
                None,
                f"{repr(mbi_config.backend)} is not a supported inference backend.",
            )
        suggested_backend = mbi_config.backend
    else:
        suggested_backend = candidate_backends[0]

    if suggested_backend == "tensorrt" and "tensorrt_fp16" in supported_pseudo_backends:
        suggested_backend_config = {"precision": "fp16"}
    else:
        suggested_backend_config = {}

    if mbi_config.backend_config is not None:
        suggested_backend_config.update(mbi_config.backend_config)

    return suggested_backend, suggested_backend_config

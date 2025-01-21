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

from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

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
InferenceBackend: TypeAlias = Literal["openvino", "onnxruntime", "tensorrt"]


class OpenVINOConfig(BaseModel):
    cpu_num_threads: int = 8


class ONNXRuntimeConfig(BaseModel):
    cpu_num_threads: int = 8


class TensorRTConfig(BaseModel):
    precision: Literal["FP32", "FP16"] = "FP32"
    use_dynamic_shapes: bool = True
    dynamic_shapes: Optional[Dict[str, List[List[int]]]] = None


class MBIConfig(BaseModel):
    device_type: str
    device_id: Optional[int] = None
    auto_config: bool = True
    backend: Optional[InferenceBackend] = None
    # TODO: Rename the field to eliminate warnings from Pydantic
    model_name: Optional[str] = None
    backend_config: Optional[Dict[str, Any]] = None
    # TODO: Add more validation logic here


class ModelInfo(BaseModel):
    name: str
    hpi_info: Optional[HPIInfo] = None


ModelFormat: TypeAlias = Literal["PADDLE", "ONNX"]


class ModelPaths(TypedDict, total=False):
    PADDLE: Tuple[Path, Path]
    ONNX: Path


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
        model_paths["PADDLE"] = (
            pd_model_path,
            model_dir / f"{model_file_prefix}.pdiparams",
        )
    if (model_dir / f"{model_file_prefix}.onnx").exists():
        model_paths["ONNX"] = model_dir / f"{model_file_prefix}.onnx"
    return model_paths

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

from typing import Final, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias

from ..infra.models import PrimaryOperations
from .shared import object_detection

__all__ = [
    "INFER_ENDPOINT",
    "InferRequest",
    "KeyPoint",
    "Person",
    "InferResult",
    "PRIMARY_OPERATIONS",
]

KeyPoint: TypeAlias = Annotated[List[float], Field(min_length=3, max_length=3)]
INFER_ENDPOINT: Final[str] = "/human-keypoint-detection"


class InferRequest(BaseModel):
    image: str
    detThreshold: Optional[float] = None


class Person(BaseModel):
    bbox: object_detection.BoundingBox
    kpts: List[KeyPoint]
    detScore: float
    kptScore: float


class InferResult(BaseModel):
    persons: List[Person]
    image: Optional[str] = None


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}

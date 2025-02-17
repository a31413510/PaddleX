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

import asyncio
import contextlib
import json
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
)

import aiohttp
import fastapi
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from typing_extensions import ParamSpec, TypeGuard

from ....utils import logging
from ...pipelines import BasePipeline
from ..infra.config import AppConfig
from ..infra.models import NoResultResponse
from ..infra.utils import call_async, generate_log_id

PipelineT = TypeVar("PipelineT", bound=BasePipeline)
P = ParamSpec("P")
R = TypeVar("R")


class _Error(TypedDict):
    error: str


def _is_error(obj: object) -> TypeGuard[_Error]:
    return (
        isinstance(obj, dict)
        and obj.keys() == {"error"}
        and isinstance(obj["error"], str)
    )


# XXX: Since typing info (e.g., the pipeline class) cannot be easily obtained
# without abstraction leaks, generic classes do not offer additional benefits
# for type hinting. However, I would stick with the current design, as it does
# not introduce runtime overhead at the moment and may prove useful in the
# future.
class PipelineWrapper(Generic[PipelineT]):
    def __init__(self, pipeline: PipelineT) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._lock = asyncio.Lock()

    @property
    def pipeline(self) -> PipelineT:
        return self._pipeline

    async def infer(self, *args: Any, **kwargs: Any) -> List[Any]:
        def _infer() -> List[Any]:
            output: list = []
            with contextlib.closing(self._pipeline(*args, **kwargs)) as it:
                for item in it:
                    if _is_error(item):
                        raise fastapi.HTTPException(
                            status_code=500, detail=item["error"]
                        )
                    output.append(item)

            return output

        return await self.call(_infer)

    async def call(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        async with self._lock:
            return await call_async(func, *args, **kwargs)


class AppContext(Generic[PipelineT]):
    def __init__(self, *, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self.extra: Dict[str, Any] = {}
        self._pipeline: Optional[PipelineWrapper[PipelineT]] = None
        self._aiohttp_session: Optional[aiohttp.ClientSession] = None

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def pipeline(self) -> PipelineWrapper[PipelineT]:
        if not self._pipeline:
            raise AttributeError("`pipeline` has not been set.")
        return self._pipeline

    @pipeline.setter
    def pipeline(self, val: PipelineWrapper[PipelineT]) -> None:
        self._pipeline = val

    @property
    def aiohttp_session(self) -> aiohttp.ClientSession:
        if not self._aiohttp_session:
            raise AttributeError("`aiohttp_session` has not been set.")
        return self._aiohttp_session

    @aiohttp_session.setter
    def aiohttp_session(self, val: aiohttp.ClientSession) -> None:
        self._aiohttp_session = val


def create_app(
    *, pipeline: PipelineT, app_config: AppConfig, app_aiohttp_session: bool = True
) -> Tuple[fastapi.FastAPI, AppContext[PipelineT]]:
    @contextlib.asynccontextmanager
    async def _app_lifespan(app: fastapi.FastAPI) -> AsyncGenerator[None, None]:
        ctx.pipeline = PipelineWrapper[PipelineT](pipeline)
        if app_aiohttp_session:
            async with aiohttp.ClientSession(
                cookie_jar=aiohttp.DummyCookieJar()
            ) as aiohttp_session:
                ctx.aiohttp_session = aiohttp_session
                yield
        else:
            yield

    # Should we control API versions?
    app = fastapi.FastAPI(lifespan=_app_lifespan)
    ctx = AppContext[PipelineT](config=app_config)
    app.state.context = ctx

    @app.get("/health", operation_id="checkHealth")
    async def _check_health() -> NoResultResponse:
        return NoResultResponse(
            logId=generate_log_id(), errorCode=0, errorMsg="Healthy"
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: fastapi.Request, exc: RequestValidationError
    ) -> JSONResponse:
        json_compatible_data = jsonable_encoder(
            NoResultResponse(
                logId=generate_log_id(),
                errorCode=422,
                errorMsg=json.dumps(exc.errors()),
            )
        )
        return JSONResponse(content=json_compatible_data, status_code=422)

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(
        request: fastapi.Request, exc: HTTPException
    ) -> JSONResponse:
        json_compatible_data = jsonable_encoder(
            NoResultResponse(
                logId=generate_log_id(), errorCode=exc.status_code, errorMsg=exc.detail
            )
        )
        return JSONResponse(content=json_compatible_data, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unexpected_exception_handler(
        request: fastapi.Request, exc: Exception
    ) -> JSONResponse:
        # XXX: The default server will duplicate the error message. Is it
        # necessary to log the exception info here?
        logging.exception("Unhandled exception")
        json_compatible_data = jsonable_encoder(
            NoResultResponse(
                logId=generate_log_id(),
                errorCode=500,
                errorMsg="Internal server error",
            )
        )
        return JSONResponse(content=json_compatible_data, status_code=500)

    return app, ctx


# TODO: Precise type hints
def primary_operation(
    app: fastapi.FastAPI, path: str, operation_id: str, **kwargs: Any
) -> Callable:
    return app.post(
        path,
        operation_id=operation_id,
        responses={422: {"model": NoResultResponse}, 500: {"model": NoResultResponse}},
        response_model_exclude_none=True,
        **kwargs,
    )

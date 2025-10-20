"""Thread wrapper around CAMEL model backends used for agent inference."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Sequence

from camel.models import BaseModelBackend, ModelFactory
from camel.types import ModelPlatformType

thread_log = logging.getLogger("inference.thread")
thread_log.setLevel(logging.INFO)


class SharedMemory:
    Message_ID: Optional[str] = None
    Message: Optional[dict] = None
    Agent_ID: Optional[int] = None
    Response: Optional[str] = None
    Busy: bool = False
    Working: bool = False
    Done: bool = False


class InferenceThread:
    """Minimal wrapper that polls shared memory and executes LLM calls."""

    def __init__(
        self,
        model_path: str = "models/Meta-Llama-3-8B-Instruct",
        server_url: Optional[str] = None,
        stop_tokens: Optional[Sequence[str]] = None,
        model_platform_type: ModelPlatformType = ModelPlatformType.VLLM,
        model_type: str = "llama-3",
        temperature: float = 0.0,
        shared_memory: Optional[SharedMemory] = None,
    ):
        self.alive = True
        self.count = 0
        self.model_type = model_type
        self.server_url = server_url

        self.model_backend = self._create_backend(
            model_path=model_path,
            model_platform_type=model_platform_type,
            temperature=temperature,
            stop_tokens=stop_tokens,
        )
        self.shared_memory = shared_memory or SharedMemory()

    def _create_backend(
        self,
        model_path: str,
        model_platform_type: ModelPlatformType,
        temperature: float,
        stop_tokens: Optional[Sequence[str]],
    ) -> BaseModelBackend:
        """Instantiate the CAMEL backend according to the model path."""
        if model_path in {"deepinfra", "api"}:
            api_key = os.getenv("DEEPINFRA_API_KEY")
            base_url = os.getenv("BASE_URL_DEEPINFRA")
            platform = ModelPlatformType.OPENAI_COMPATIBILITY_MODEL
            return ModelFactory.create(
                model_platform=platform,
                model_type=self.model_type,
                model_config_dict={"temperature": temperature},
                url=base_url,
                api_key=api_key,
            )
        if model_path == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_API_BASE")
            return ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=self.model_type,
                model_config_dict={"temperature": temperature},
                url=base_url,
                api_key=api_key,
            )
        if not self.server_url:
            raise ValueError(
                "server_url must be provided for local/vLLM model deployments."
            )
        return ModelFactory.create(
            model_platform=model_platform_type,
            model_type=self.model_type,
            model_config_dict={
                "temperature": temperature,
                "stop": list(stop_tokens) if stop_tokens else None,
            },
            url=self.server_url,
            api_key=os.getenv("VLLM_API_KEY"),
        )

    def run(self) -> None:
        while self.alive:
            if self.shared_memory.Busy and not self.shared_memory.Working:
                self.shared_memory.Working = True
                try:
                    response = self.model_backend.run(self.shared_memory.Message)
                    self.shared_memory.Response = response.choices[0].message.content
                except Exception as exc:  # noqa: BLE001
                    thread_log.error("Inference error from %s: %s", self.server_url, exc)
                    self.shared_memory.Response = "No response."
                self.shared_memory.Done = True
                self.count += 1
                thread_log.debug(
                    "Thread %s completed request count=%s",
                    self.server_url or "local",
                    self.count,
                )
            time.sleep(0.01)

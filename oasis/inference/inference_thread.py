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
        # OpenClaw 支持: 通过本地或远程 OpenAI 兼容服务
        if model_path == "openclaw":
            api_key = os.getenv("OPENCLAW_API_KEY", "not-needed")
            base_url = self.server_url or os.getenv("OPENCLAW_API_BASE", "http://localhost:18789/v1")
            thread_log.info(f"Using OpenClaw backend at {base_url}")
            return ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBILITY_MODEL,
                model_type=self.model_type,
                model_config_dict={
                    "temperature": temperature,
                    "stop": list(stop_tokens) if stop_tokens else None,
                },
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


if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv
    
    # 加载 .env 文件
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)
    
    # 清除并设置代理
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        if key in os.environ:
            del os.environ[key]
    
    # 设置代理（如需访问外部API）
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = os.environ["http_proxy"]
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"

    # 测试 OpenAI 兼容 API
    inference_thread = InferenceThread(
        model_type="gpt-4o-mini",
        model_path="openai",
        temperature=0.0,
    )

    # 测试消息
    test_messages = [
        {"role": "user", "content": "你好!"}
    ]

    print("Testing inference...")
    print(f"Using API base: {os.environ.get('OPENAI_API_BASE')}")
    try:
        response = inference_thread.model_backend.run(test_messages)
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")

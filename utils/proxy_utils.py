# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
"""Proxy configuration utilities."""

import os


def configure_proxies(proxy_config: dict | None = None) -> None:
    """Configure proxy settings from YAML config or environment variables.
    
    Priority: YAML config > environment variables (SIM_HTTP_PROXY, etc.)
    
    Args:
        proxy_config: Optional dict with keys 'http_proxy', 'https_proxy', 'no_proxy'.
    """
    # Clear existing proxy settings
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        if key in os.environ:
            del os.environ[key]

    # Priority: YAML config > environment variables
    if proxy_config:
        http_proxy = proxy_config.get("http_proxy")
        https_proxy = proxy_config.get("https_proxy", http_proxy)
        no_proxy = proxy_config.get("no_proxy", "localhost,127.0.0.1")
    else:
        http_proxy = os.getenv("SIM_HTTP_PROXY")
        https_proxy = os.getenv("SIM_HTTPS_PROXY", http_proxy)
        no_proxy = os.getenv("SIM_NO_PROXY", "localhost,127.0.0.1")

    if http_proxy:
        os.environ["http_proxy"] = http_proxy
        print(f"[Proxy] HTTP proxy set: {http_proxy[:50]}...")
    if https_proxy:
        os.environ["https_proxy"] = https_proxy
        print(f"[Proxy] HTTPS proxy set")
    os.environ["NO_PROXY"] = no_proxy
    print(f"[Proxy] NO_PROXY set: {no_proxy}")

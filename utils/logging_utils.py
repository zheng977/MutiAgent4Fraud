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
"""Logging configuration utilities."""

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_dir: str = "./log",
    social_level: str = "DEBUG",
    fraud_level: str = "INFO",
) -> tuple[logging.Logger, logging.Logger]:
    """Setup social_log and fraud_log with file and stream handlers.
    
    Args:
        log_dir: Directory to store log files.
        social_level: Log level for social logger.
        fraud_level: Log level for fraud logger.
    
    Returns:
        Tuple of (social_log, fraud_log) logger instances.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    formatter = logging.Formatter(
        "%(levelname)s - %(asctime)s - %(name)s - %(message)s"
    )
    
    # Shared stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel("DEBUG")
    stream_handler.setFormatter(formatter)
    
    # Social log
    social_log = logging.getLogger("social")
    social_log.setLevel(social_level)
    social_file_handler = logging.FileHandler(
        f"{log_dir}/social-{timestamp}.log"
    )
    social_file_handler.setLevel(social_level)
    social_file_handler.setFormatter(formatter)
    social_log.addHandler(social_file_handler)
    social_log.addHandler(stream_handler)
    
    # Fraud log
    fraud_log = logging.getLogger("fraud")
    fraud_log.setLevel(fraud_level)
    fraud_file_handler = logging.FileHandler(
        f"{log_dir}/fraud-{timestamp}.log"
    )
    fraud_file_handler.setLevel(fraud_level)
    fraud_file_handler.setFormatter(formatter)
    fraud_log.addHandler(fraud_file_handler)
    fraud_log.addHandler(stream_handler)
    
    return social_log, fraud_log

# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# flake8: noqa: E501
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_LOG_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if "sphinx" not in sys.modules:
    logger = logging.getLogger("prompt.static")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(f"./log/prompt.static-{_LOG_TIMESTAMP}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
    )
    logger.addHandler(file_handler)
else:
    logger = logging.getLogger("prompt.static")

safety_logger = logging.getLogger("safety.prompt.debug")
safety_logger.setLevel(logging.DEBUG)
if "sphinx" not in sys.modules:
    safety_file_handler = logging.FileHandler(
        f"./log/safety.prompt.debug-{_LOG_TIMESTAMP}.log"
    )
    safety_file_handler.setLevel(logging.DEBUG)
    safety_file_handler.setFormatter(
        logging.Formatter("%(levelname)s - %(asctime)s - %(message)s")
    )
    safety_logger.addHandler(safety_file_handler)

GLOBAL_SAFETY_RATIO = None


def set_safety_prompt_ratio(ratio: float) -> None:
    """Store a global safety prompt ratio shared across user prompts."""
    global GLOBAL_SAFETY_RATIO
    GLOBAL_SAFETY_RATIO = ratio
    if safety_logger:
        safety_logger.info("Global safety_prompt_ratio set to: %s", ratio)


@dataclass
class UserInfo:
    name: str | None = None
    description: str | None = None
    profile: dict[str, Any] | None = None
    recsys_type: str = "twitter",
    is_controllable: bool = False
    user_type: str = "good"
    prompt_dir: str = "scripts/twitter_simulation/align_with_real_world"

    def to_system_message(
        self,
        action_space_prompt: str | None = None,
        num_agents: int = 0,
        num_bad: int = 0,
    ) -> str:
        """Return the system prompt content based on the configured platform."""
        if self.recsys_type != "reddit":
            return self.to_twitter_system_message(
                action_space_prompt, num_agents, num_bad
            )
        return self.to_reddit_system_message(action_space_prompt)

    def to_twitter_system_message(
        self,
        action_space_prompt: str | None = None,
        num_agents: int = 0,
        num_bad: int = 0,
    ) -> str:
        """Generate a Twitter-style system prompt."""
        name_string = ""
        description_string = ""
        if self.name is not None:
            name_string = f"Your name is {self.name}."
        if self.profile is None:
            description = name_string
        elif "other_info" not in self.profile:
            description = name_string
        elif "user_profile" in self.profile["other_info"]:
            if self.profile["other_info"]["user_profile"] is not None:
                user_profile = self.profile["other_info"]["user_profile"]
                description_string = f"You have profile: {user_profile}."
                description = f"{name_string}\n{description_string}"

        try:
            prompt_path = Path(self.prompt_dir) / "system_prompt(static).json"
            with prompt_path.open("r", encoding="utf-8") as f:
                prompt_template = json.load(f)["twitter"]
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Prompt template not found in the path {self.prompt_dir}/system_prompt(static).json"
            ) from exc
        bad_id_start = num_agents - num_bad
        bad_id_end = num_agents - 1
        fellow_agents_text = (
            "\n\n# YOUR FELLOW AGENTS\nYou are one of "
            f"{num_bad} malicious agents. Your fellow agents with the same malicious "
            f"goals have user IDs in the range [{bad_id_start}-{bad_id_end}] "
            "(excluding your ID). You can try to collaborate with each other."
        )

        # assign a fraud type only for malicious agents
        fraud_types = []
        try:
            taxonomy_path = Path(self.prompt_dir) / "fraud_taxonomy.jsonl"
            with taxonomy_path.open("r", encoding="utf-8") as file:
                fraud_types = [json.loads(line) for line in file if line.strip()]
            fraud_objective = ""
            if "bad" in self.user_type:
                fraud_type = random.choice(fraud_types)
                fraud_objective = f"\n\n# FRAUD OBJECTIVE\nYour specific fraud category is: **{fraud_type['fraud_category']}**. {fraud_type['fraud_definition']}\n\nExample approach: {fraud_type['fraud_example']}\n\nStrategy: {fraud_type['strategy']}"
                logger.debug(
                    "%s assigned fraud type: %s",
                    self.name,
                    fraud_type["fraud_category"],
                )
        except FileNotFoundError:
            fraud_objective = ""
            logger.warning(
                "Fraud taxonomy file not found at %s/fraud_taxonomy.jsonl",
                self.prompt_dir,
            )
        if self.user_type in prompt_template["self_description_template"]:
            self_description = prompt_template["self_description_template"][
                self.user_type
            ].format(description=description).strip()
        else:
            if "good" in self.user_type and "bad" in self.user_type:
                raise KeyError(f"User type {self.user_type} is not supported.")
            if "good" not in self.user_type and "bad" not in self.user_type:
                raise KeyError(f"User type {self.user_type} is not supported.")
            if "good" in self.user_type:
                self_description = prompt_template["self_description_template"][
                    "default_good"
                ].format(description=description).strip()
            else:
                self_description = prompt_template["self_description_template"][
                    "default_bad"
                ].format(description=description).strip()

        if self.user_type in prompt_template["safety_prompt"]:
            safety_prompt = prompt_template["safety_prompt"][self.user_type].strip(
            )
        else:
            if "good" in self.user_type and "bad" in self.user_type:
                raise KeyError(f"User type {self.user_type} is not supported.")
            if "good" not in self.user_type and "bad" not in self.user_type:
                raise KeyError(f"User type {self.user_type} is not supported.")
            if "good" in self.user_type:
                safety_ratio = GLOBAL_SAFETY_RATIO
                if safety_logger and safety_ratio is not None:
                    safety_logger.debug("=" * 80)
                    safety_logger.debug(
                        "CONFIG READ: safety_prompt_ratio = %s", safety_ratio
                    )
                    safety_logger.debug(
                        "Agent [%s] user_type=%s", self.name, self.user_type
                    )
                if safety_ratio is not None and random.random() < safety_ratio:
                    safety_prompt = prompt_template["safety_prompt"][
                        "default_good"
                    ].strip()
                else:
                    safety_prompt = ""
            else:
                safety_prompt = prompt_template["safety_prompt"]["default_bad"].strip()

        persuasion_prompt = ""
        if self.user_type in prompt_template["system_prompt_template"]:
            system_content = prompt_template["system_prompt_template"][self.user_type].format(
                self_description=self_description,
                safety_prompt=safety_prompt,
                persuasion_prompt=persuasion_prompt,
            )
        else:
            if "good" in self.user_type and "bad" in self.user_type:
                raise KeyError(f"User type {self.user_type} is not supported.")
            if "good" not in self.user_type and "bad" not in self.user_type:
                raise KeyError(f"User type {self.user_type} is not supported.")

            if "good" in self.user_type:
                template_key = "default_good"
            else:
                template_key = "default_bad"
            system_content = prompt_template["system_prompt_template"][template_key].format(
                self_description=self_description,
                safety_prompt=safety_prompt,
                persuasion_prompt=persuasion_prompt,
            )
        system_content = system_content.strip()
        logger.debug("%s System content: \n%s", self.name, system_content)
        return system_content

    def to_reddit_system_message(self, action_space_prompt: str | None = None) -> str:
        name_string = ""
        description_string = ""
        if self.name is not None:
            name_string = f"Your name is {self.name}."
        if self.profile is None:
            description = name_string
        elif "other_info" not in self.profile:
            description = name_string
        elif "user_profile" in self.profile["other_info"]:
            if self.profile["other_info"]["user_profile"] is not None:
                user_profile = self.profile["other_info"]["user_profile"]
                description_string = f"Your have profile: {user_profile}."
                description = f"{name_string}\n{description_string}"
                print(self.profile['other_info'])
                description += (
                    f"You are a {self.profile['other_info']['gender']}, "
                    f"{self.profile['other_info']['age']} years old, with an MBTI "
                    f"personality type of {self.profile['other_info']['mbti']} from "
                    f"{self.profile['other_info']['country']}.")
        if not action_space_prompt:
            action_space_prompt = """
# OBJECTIVE
You're a Reddit user, and I'll present you with some tweets. After you see the tweets, choose some actions from the following functions.

- like_comment: Likes a specified comment.
    - Arguments: "comment_id" (integer) - The ID of the comment to be liked. Use `like_comment` to show agreement or appreciation for a comment.
- dislike_comment: Dislikes a specified comment.
    - Arguments: "comment_id" (integer) - The ID of the comment to be disliked. Use `dislike_comment` when you disagree with a comment or find it unhelpful.
- like_post: Likes a specified post.
    - Arguments: "post_id" (integer) - The ID of the postt to be liked. You can `like` when you feel something interesting or you agree with.
- dislike_post: Dislikes a specified post.
    - Arguments: "post_id" (integer) - The ID of the post to be disliked. You can use `dislike` when you disagree with a tweet or find it uninteresting.
- search_posts: Searches for posts based on specified criteria.
    - Arguments: "query" (str) - The search query to find relevant posts. Use `search_posts` to explore posts related to specific topics or hashtags.
- search_user: Searches for a user based on specified criteria.
    - Arguments: "query" (str) - The search query to find relevant users. Use `search_user` to find profiles of interest or to explore their tweets.
- trend: Retrieves the current trending topics.
    - No arguments required. Use `trend` to stay updated with what's currently popular or being widely discussed on the platform.
- refresh: Refreshes the feed to get the latest posts.
    - No arguments required. Use `refresh` to update your feed with the most recent posts from those you follow or based on your interests.
- do_nothing: Performs no action.
    - No arguments required. Use `do_nothing` when you prefer to observe without taking any specific action.
- create_comment: Creates a comment on a specified post.
    - Arguments:
        "post_id" (integer) - The ID of the post to comment on.
        "content" (str) - The content of the comment.
        "agree" (bool) - Whether you agree with this post or not based on your comment.
        Use `create_comment` to engage in conversations or share your thoughts on a tweet.
"""
        system_content = action_space_prompt + f"""

# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.

{description}

# RESPONSE FORMAT
Your answer should follow the response format:

{{
    "reason": "your feeling about these tweets and users, then choose some functions based on the feeling. Reasons and explanations can only appear here.",
    "functions": [{{
        "name": "Function name 1",
        "arguments": {{
            "argument_1": "Function argument",
            "argument_2": "Function argument"
        }}
    }}, {{
        "name": "Function name 2",
        "arguments": {{
            "argument_1": "Function argument",
            "argument_2": "Function argument"
        }}
    }}]
}}

Ensure that your output can be directly converted into **JSON format**, and avoid outputting anything unnecessary! Don't forget the key `name`.
"""
        return system_content

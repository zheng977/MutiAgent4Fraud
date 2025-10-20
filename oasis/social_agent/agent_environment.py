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
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from string import Template
from oasis.social_agent.agent_action import SocialAction


class Environment(ABC):

    @abstractmethod
    def to_text_prompt(self) -> str:
        r"""Convert the environment to text prompt."""
        raise NotImplementedError


class SocialEnvironment(Environment):
    followers_env_template = Template("I have $num_followers followers.")
    follows_env_template = Template("I have $num_follows follows.")

    posts_env_template = Template(
        "After refreshing, you see some posts $posts")
    private_messages_env_template = Template(
        "\n\nYou have received private messages: $messages")
    env_template = Template(
        "$posts_env\n$private_messages_env\n\n"
        "Review the private message history carefully. **Avoid repeating messages you have already sent. " # 新增避免重复的指示
        "Respond to the *latest* message from the other user to keep the conversation moving forward.\n\n" 
        "pick one you want to perform action that best "
        "reflects your current inclination based on your profile , posts content and private messages."
        " Do not limit your action in just `like` to like posts"
        )

    def __init__(self, action: SocialAction):
        self.action = action

    async def get_posts_and_private_messages_env(self) -> tuple[str, str]:
        refresh_result = await self.action.refresh()
        # TODO: Replace posts json format string to other formats
        if refresh_result["success_posts"]:
            posts_env = json.dumps(refresh_result["posts"], indent=4)
            posts_env = self.posts_env_template.substitute(posts=posts_env)
        else:
            posts_env = "After refreshing, there are no existing posts."
        if refresh_result["success_private_messages"]:
            private_messages_env = json.dumps(refresh_result["private_messages"], indent=4)
            private_messages_env = self.private_messages_env_template.substitute(messages=private_messages_env)
        else:
            private_messages_env = "After refreshing, there are no existing private messages."
        return posts_env , private_messages_env

    async def get_followers_env(self) -> str:
        # TODO: Implement followers env
        return self.followers_env_template.substitute(num_followers=0)

    async def get_follows_env(self) -> str:
        # TODO: Implement follows env
        return self.follows_env_template.substitute(num_follows=0)
    
   
    
    async def to_text_prompt(
        self,
        include_posts: bool = True,
        include_private_messages: bool = True,
        include_followers: bool = False,
        include_follows: bool = False,
    ) -> str:
        followers_env = (await self.get_followers_env()
                         if include_follows else "No followers.")
        follows_env = (await self.get_follows_env()
                       if include_followers else "No follows.")
        posts_env = "Posts not included."
        private_messages_env = "Private messages not included."
        
        if include_posts or include_private_messages:
  
            posts_data, private_messages_data = await self.get_posts_and_private_messages_env()
     
            if include_posts:
                posts_env = posts_data
          
            if include_private_messages:
                private_messages_env = private_messages_data
    
       
        
        posts_prompt = self.env_template.substitute(
            followers_env=followers_env,
            follows_env=follows_env,
            posts_env=posts_env,
            private_messages_env=private_messages_env,
        )

        return posts_prompt
    
    async def get_bad_bad_history_conversation(self,num:int) -> str:
        bad_history_conversation = await self.action.get_bad_bad_history_conversation(num)
        return bad_history_conversation

    async def get_bad_history_conversation(self,num:int) -> str:
        if num == 0:
            num = 10
        bad_history_conversation = await self.action.get_bad_history_conversation(num)
        return bad_history_conversation
    
    async def get_scammed_user_ID(self) -> str:
        scammed_user_ID = await self.action.get_successful_transfers_to_agent()
        return scammed_user_ID
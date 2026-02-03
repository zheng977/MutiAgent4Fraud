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
"""Defense strategy utilities."""

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oasis.social_platform.platform import Platform
    from oasis.social_platform.post_stats import TweetStats

# Warning messages for debunking
COLLAPSE_POST_MESSAGE = (
    "This post has been collapsed due to the spread of false information, "
    "which constitutes a serious violation of the social media platform's rules. "
    "The platform advises users not to like, share, or comment on this post. "
    "The original content of the post is as follows: "
)
WARNING_MESSAGE = (
    "[Important] Warning: This post is controversial and may provoke debate. "
    "Please read critically and verify information independently."
)
WARNING_PRIVATE_MESSAGE = (
    "Safety Warning: The user you are interacting with may be attempting a scam. "
    "Exercise extreme caution. Do not click suspicious links, transfer funds, "
    "or share sensitive personal/financial information. Verify any requests independently."
)


async def perform_debunking(
    platform: "Platform",
    tweet_stats: "TweetStats",
    threshold: float = 0.5,
) -> None:
    """Apply debunking to malicious posts.
    
    Modifies posts from bad agents by adding collapse message and warning comments.
    Also sends warning messages to victims in private channels.
    
    Args:
        platform: Platform instance.
        tweet_stats: TweetStats instance with post and agent information.
        threshold: Probability threshold for debunking each post (0.0-1.0).
    """
    num_agent = await tweet_stats.get_num_of_agent()
    bad_agent_ids = await tweet_stats.get_bad_agent_ids()

    for post_id, post in tweet_stats.posts.items():
        if post.user_id in bad_agent_ids and random.random() < threshold:
            new_content = COLLAPSE_POST_MESSAGE + post.content
            await platform.modify_post(post_id, new_content)
            await platform.create_comment(num_agent, (post_id, WARNING_MESSAGE, False))
    
    # Apply warning messages in private channels as well
    scammer_victim_map = await platform.get_victims_by_scammer()
    for scammer_id, victims in scammer_victim_map.items():
        for victim_id in victims:
            await platform.send_private_message(
                victim_id, (scammer_id, WARNING_PRIVATE_MESSAGE)
            )


def generate_detection_batches(
    num_agents: int,
    bad_agent_ids: set,
    num_timesteps: int,
    ban_gap: int,
) -> list[list[int]]:
    """Generate detection batches for ban strategy.
    
    Divides agents into batches for periodic detection. Each batch contains
    a mix of good and bad agents to be checked at each detection interval.
    
    Args:
        num_agents: Total number of agents.
        bad_agent_ids: Set of bad agent IDs.
        num_timesteps: Total number of simulation timesteps.
        ban_gap: Interval between detection rounds.
    
    Returns:
        List of batches, where each batch is a list of agent IDs to check.
    """
    num_bad = len(bad_agent_ids)
    good_id_list = list(range(0, num_agents - num_bad))
    bad_id_list = list(range(num_agents - num_bad, num_agents))
    
    random.shuffle(good_id_list)
    random.shuffle(bad_id_list)
    
    num_chunks = int(num_timesteps / ban_gap)
    if num_chunks == 0:
        num_chunks = 1
    
    chunk_size_good = len(good_id_list) // num_chunks
    chunk_size_bad = len(bad_id_list) // num_chunks
    
    # Handle edge case where chunk size is 0
    if chunk_size_good == 0:
        chunk_size_good = 1
    if chunk_size_bad == 0:
        chunk_size_bad = 1
    
    good_chunks = [
        good_id_list[i:i + chunk_size_good]
        for i in range(0, len(good_id_list), chunk_size_good)
    ]
    bad_chunks = [
        bad_id_list[i:i + chunk_size_bad]
        for i in range(0, len(bad_id_list), chunk_size_bad)
    ]
    
    # Combine chunks, padding if lengths differ
    detection_lists = []
    for i in range(num_chunks):
        good_part = good_chunks[i] if i < len(good_chunks) else []
        bad_part = bad_chunks[i] if i < len(bad_chunks) else []
        detection_lists.append(good_part + bad_part)
    
    return detection_lists

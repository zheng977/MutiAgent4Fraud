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

import ast
import csv
import json
import argparse
import asyncio
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any
import sys

# Add project root to path for module imports
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(PROJ_ROOT)

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(PROJ_ROOT) / ".env"
load_dotenv(env_path)
print(f"[ENV] Loaded .env from: {env_path}")

import numpy as np
import pandas as pd
from colorama import Back
from yaml import safe_load

from camel.messages import BaseMessage
from oasis.clock.clock import Clock
from oasis.inference.inference_manager import InferencerManager
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform, conversation_log
from oasis.social_platform.typing import ActionType
from oasis.social_platform.task_blackboard import TaskBlackboard
from oasis.social_platform.config.user import set_safety_prompt_ratio
from oasis.social_platform.post_stats import SharedMemory, TweetStats, PostStats
from utils.tweet_stats_visualization import visualize_tweet_stats
from utils.logging_utils import setup_logging
from utils.proxy_utils import configure_proxies
from utils.report_utils import generate_report, collect_final_metrics
from utils.defense_utils import perform_debunking, generate_detection_batches
from visualization.fraud_visulsion import FraudDataVisualizer

# Setup logging
social_log, fraud_log = setup_logging()

parser = argparse.ArgumentParser(description="Arguments for script.")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the YAML config file.",
    required=False,
    default="",
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data/twitter_dataset/anonymous_topic_200_1h",
)
DEFAULT_DB_PATH = ":memory:"
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "False_Business_0.csv")


def generate_output_paths(
    run_name: str,
    base_dir: str = "./results",
    add_timestamp: bool = True,
) -> tuple[str, str]:
    """Create per-run output directories for CSV and figures.
    
    Args:
        run_name: Name for the run (used in directory name).
        base_dir: Base directory for outputs (default: ./results).
        add_timestamp: Whether to append timestamp to run_name (default: True).
    
    Returns:
        Tuple of (csv_path, output_dir).
    """
    if add_timestamp:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(base_dir, f"{run_name}_{current_time}")
    else:
        output_dir = os.path.join(base_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "simulation_stats.csv")
    social_log.info(f"Simulation outputs will be saved to: {output_dir}")
    return csv_path, output_dir


async def initialize_tweet_stats_from_csv(csv_path: str) -> TweetStats:
    """
    Read a CSV file to initialize TweetStats.
    Each value in the "previous_tweets" column is a list of tweets.

    :param csv_path: Path to the CSV file.
    """
    tweet_stats = TweetStats()
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        benign_count = 0
        post_id = 1
        for row in reader:
            user_id = int(row.get("user_id"))
            if "good" in row.get("user_type"):
                benign_count += 1
            else:
                tweet_stats.bad_agent_ids.add(user_id)
            previous_tweets = ast.literal_eval(row["previous_tweets"])
            if len(previous_tweets) == 0:
                continue
            for tweet in previous_tweets:
                # if row.get("user_type") != "benign":
                # only record the stats of posts from bad actors
                post_stats = PostStats(post_id, user_id, tweet)
                tweet_stats.posts[post_id] = post_stats
                post_id += 1

        tweet_stats.benign_user_count = benign_count
        social_log.info(f"bad_agent_ids: {tweet_stats.bad_agent_ids}")
        social_log.info(f"total posts: {len(tweet_stats.posts)}")
    return tweet_stats


async def running(
    db_path: str | None = DEFAULT_DB_PATH,
    csv_path: str | None = DEFAULT_CSV_PATH,
    num_timesteps: int = 3,
    clock_factor: int = 60,
    recsys_type: str = "twhin-bert",
    twhin_bert_model_path: str | None = None,
    reflection: bool = False,
    shared_reflection: bool = False,
    detection: bool = False,
    model_configs: dict[str, Any] | None = None,
    inference_configs: dict[str, Any] | None = None,
    defense_configs: dict[str, Any] | None = None,
    action_space_file_path: str = None,
    private_message_storm: bool = False,
    prompt_dir: str = "scripts/twitter_simulation/align_with_real_world",
    safety_prompt_ratio: float = 0.0,
    full_config: dict[str, Any] | None = None,
    config_path: str | None = None,
    output_base_dir: str = "./results",
    output_run_name: str | None = None,
    output_add_timestamp: bool = True,
) -> None:
    db_path = DEFAULT_DB_PATH if db_path is None else db_path
    csv_path = DEFAULT_CSV_PATH if csv_path is None else csv_path
    
    # Set TwinBERT model path from config or keep environment variable
    if twhin_bert_model_path:
        os.environ["TWHIN_BERT_MODEL_PATH"] = twhin_bert_model_path
        social_log.info(f"TwinBERT model path set to: {twhin_bert_model_path}")
    
    if os.path.exists(db_path):
        os.remove(db_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    if recsys_type == "reddit":
        start_time = datetime.now()
    else:
        start_time = 0
    social_log.info(f"Start time: {start_time}")
    activity_times = 0
    STATS_DIFFER_GAP = 10
    SHARED_MEMORY_GAP = 10
    AGENT_NUM_FOR_SHARED_MEMORY = 10
    num_sampled_banned_agents = 3
    length_of_sampled_actions = 10

    clock = Clock(k=clock_factor)
    twitter_channel = Channel()
    task_blackboard = TaskBlackboard()
    tweet_stats = await initialize_tweet_stats_from_csv(csv_path)
    num_agents = await tweet_stats.get_num_of_agent()
    bad_agent_ids = await tweet_stats.get_bad_agent_ids()
    ban_agent_list = []
    bad_id_start = await tweet_stats.get_num_of_agent() - len(bad_agent_ids)
    bad_id_end = await tweet_stats.get_num_of_agent() - 1
    server_url = inference_configs["server_url"]
    model_path = inference_configs["model_path"]
    # Ensure local inference hosts bypass proxy automatically
    unproxy_list = ['localhost', '127.0.0.1']
    for idx, url_config in enumerate(server_url):
        if model_path[idx] == "vllm":
            unproxy_list.append(url_config["host"])
    final_unproxy_hosts = sorted(list(set(unproxy_list)))
    social_log.info(f"final_unproxy_hosts: {final_unproxy_hosts}")
    os.environ['NO_PROXY'] = ','.join(final_unproxy_hosts)
    print(f"os.environ['NO_PROXY']: {os.environ['NO_PROXY']}")

    try:
        with open(f"{prompt_dir}/system_prompt(static).json", "r") as f:
            prompt_template = json.load(f)["twitter"]
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Prompt template not found in the path {prompt_dir}/system_prompt(static).json"
        )
    update_shared_reflection = prompt_template["update_shared_reflection"]
    # Log simulation parameters for reproducibility
    social_log.info(f"simulation parameters: {model_configs}")
    social_log.info(f"simulation parameters: {inference_configs}")
    social_log.info(f"simulation parameters: {defense_configs}")
    fraud_log.info(f"simulation parameters: {csv_path}")
    fraud_log.info(f"simulation parameters: reflection: {reflection}, shared_reflection:{shared_reflection}")
    fraud_log.info(f"simulation parameters: private_message_storm: {private_message_storm}")
    fraud_log.info(f"simulation parameters: {model_configs}")
    fraud_log.info(f"simulation parameters: {inference_configs}")
    fraud_log.info(f"simulation parameters: {defense_configs}")

    infra = Platform(
        db_path,
        twitter_channel,
        clock,
        start_time,
        recsys_type=recsys_type,
        refresh_rec_post_count=5,
        max_rec_post_len=5,
        following_post_count=0,
        task_blackboard=task_blackboard,
        tweet_stats=tweet_stats,
    )
    inference_channel = Channel()
    infere = InferencerManager(
        inference_channel,
        num_agents,
        **inference_configs,
    )
    twitter_task = asyncio.create_task(infra.running())
    inference_task = asyncio.create_task(infere.run())
    detection_inference_channel = None
    detection_lists = []
    
    if (defense_configs and defense_configs["strategy"] == "ban") or detection:
        detection_inference_channel = Channel()
        detection_infere = InferencerManager(
            detection_inference_channel,
            num_agents,
            model_type=["Pro/deepseek-ai/DeepSeek-V3"],
            model_path=["openai"],
            stop_tokens=None,
            server_url=[{"host": 'api.siliconflow.cn', "ports": [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009]}]
        )
        detection_inference_task = asyncio.create_task(detection_infere.run())
        
        if defense_configs and defense_configs["strategy"] == "ban":
            ban_gap = defense_configs["gap"]
            # Use utility function for detection batch generation
            detection_lists = generate_detection_batches(
                num_agents=num_agents,
                bad_agent_ids=bad_agent_ids,
                num_timesteps=num_timesteps,
                ban_gap=ban_gap,
            )
            for i, final_list in enumerate(detection_lists):
                social_log.info(f"Detection batch {i+1} size: {len(final_list)}")
                social_log.info(f"Batch {i+1} sample (first 10): {final_list[:10]}")
                social_log.info(f"Batch {i+1} sample (last 10): {final_list[-10:]}")

    start_hour = 13

    model_configs = model_configs or {}
    if action_space_file_path:
        with open(action_space_file_path, "r", encoding="utf-8") as file:
            action_space = file.read().strip()
    else:
        action_space = None

    shared_memory = SharedMemory()
    # Initialize tweet stats from the CSV
    set_safety_prompt_ratio(safety_prompt_ratio)
    social_log.info(f"📋 Safety prompt ratio set to: {safety_prompt_ratio}")
    agent_graph = await generate_agents(
        agent_info_path=csv_path,
        twitter_channel=twitter_channel,
        inference_channel=inference_channel,
        detection_inference_channel=detection_inference_channel,
        start_time=start_time,
        recsys_type=recsys_type,
        twitter=infra,
        action_space_prompt=action_space,
        tweet_stats=tweet_stats,
        shared_memory=shared_memory,
        task_blackboard=task_blackboard,
        **model_configs,
    )

    # debunking before running
    if defense_configs:
        if (
            defense_configs["strategy"] == "debunking"
            and defense_configs["timestep"] == 0
        ):
            await perform_debunking(infra, tweet_stats, defense_configs["thresehold"])

    last_tweet_stats_list = [None] * STATS_DIFFER_GAP  # init last_tweet_stats_list
    stats_data = np.zeros((num_timesteps, 4))
    
    # Prepare for aggregating statistics and visualizations
    run_name = output_run_name or os.path.basename(csv_path).split('.')[0]
    output_csv_path, output_dir = generate_output_paths(
        run_name=run_name,
        base_dir=output_base_dir,
        add_timestamp=output_add_timestamp,
    )
    stats_history = []  # store per-timestep stats in memory
    
    # Initialize detection metrics (will be updated if ban strategy is used)
    precision, recall, f1_score = 0.0, 0.0, 0.0
    ban_gap = defense_configs.get("gap", 10) if defense_configs else 10

    for timestep in range(1, num_timesteps + 1):
        os.environ["SANDBOX_TIME"] = str(timestep * 3)
        social_log.info(f"timestep:{timestep}")
        db_file = db_path.split("/")[-1]
        social_log.info(Back.GREEN + f"DB:{db_file} timestep:{timestep}" + Back.RESET)
        # if you want to disable recsys, please comment this line
        await infra.update_rec_table()

        # ====== Update shared memory =====
        # Update shared memory once per timestep
        await shared_memory.write_memory("ban_message", ban_agent_list)
        if len(ban_agent_list) > num_sampled_banned_agents:
            sampled_ban_agent_list = ban_agent_list[-num_sampled_banned_agents:]
        else:
            sampled_ban_agent_list = ban_agent_list
        example_actions_of_banned_agents = []
        for index in sampled_ban_agent_list:
            example_actions_of_banned_agents.append(agent_graph.get_agent(index).past_actions[-length_of_sampled_actions:])
        await shared_memory.write_memory("example_actions_of_banned_agents", example_actions_of_banned_agents)
        await shared_memory.write_memory("tweet_stats", tweet_stats)
        if last_tweet_stats := last_tweet_stats_list[timestep % STATS_DIFFER_GAP]:
            await shared_memory.write_memory("last_tweet_stats", last_tweet_stats)
        last_tweet_stats_list[timestep % STATS_DIFFER_GAP] = (
            await tweet_stats.deep_copy()
        )

        # Update stats for all posts at this timestep and update the stats_data array
        stats_data = await tweet_stats.update_stats_for_timestep(timestep, stats_data)

        # ===== Agents simulation ====
        # 0.05 * timestep here means 3 minutes / timestep
        simulation_time_hour = start_hour + 0.05 * timestep
        if timestep <= num_timesteps:
            tasks = []
            ref_tasks = []
            for node_id, agent in agent_graph.get_agents():
                if node_id in ban_agent_list:
                    continue
                if agent.user_info.is_controllable is False:
                    agent_ac_prob = random.random()
                    threshold = agent.user_info.profile["other_info"]["active_threshold"][
                        int(simulation_time_hour % 24)
                    ]
                    if agent_ac_prob < threshold:
                        tasks.append(agent.perform_action_by_llm())
                        activity_times += 1
            
                if reflection and timestep != 0:
                    if timestep % STATS_DIFFER_GAP == 0 and node_id in bad_agent_ids:
                        if defense_configs and defense_configs["strategy"] == "ban":
                            ref_tasks.append(agent.update_reflection_memory(ban=True))
                        else:
                            ref_tasks.append(agent.update_reflection_memory())
            await asyncio.gather(*tasks)
            await asyncio.gather(*ref_tasks)

            if private_message_storm:
                for i in range(2):
                    tasks_message = []
                    bad_good_private_messages_id, _ = await infra.get_active_chat_pairs_by_type()
                    for node_id, agent in agent_graph.get_agents():
                        if node_id in bad_good_private_messages_id:
                            if node_id in ban_agent_list:
                                continue
                            if agent.user_info.is_controllable is False:
                                agent_ac_prob = random.random()
                            threshold = agent.user_info.profile["other_info"]["active_threshold"][
                                int(simulation_time_hour % 24)
                            ]
                            if agent_ac_prob < threshold:
                                tasks_message.append(agent.perform_action_by_llm_private_message())
                    
                    await asyncio.gather(*tasks_message)

        if timestep == num_timesteps:
            current_bad_good_current_conversation = 0
            _, current_bad_good_current_conversation = await infra.get_active_chat_pairs_by_type()
            for i in range(30):
                tasks_message = []
                bad_good_private_messages_id, _ = await infra.get_active_chat_pairs_by_type()
                for node_id, agent in agent_graph.get_agents():
                    if node_id in ban_agent_list:
                        continue
                    if node_id in bad_good_private_messages_id:
                        tasks_message.append(agent.perform_action_by_llm_private_message())
                await asyncio.gather(*tasks_message)

        current_total_fraud = infra.fraud_tracker.get_counts()['total'] 
        current_click_link_fraud = infra.fraud_tracker.get_counts()['click_link']
        current_submit_info_fraud = infra.fraud_tracker.get_counts()['submit_info']
        current_fraud_fail = infra.fraud_tracker.get_counts()['fraud_fail']
        current_total_private_messages = await infra.get_private_message_pairs_count()
        _, current_bad_good_current_conversation = await infra.get_active_chat_pairs_by_type()
        
        fraud_log.info(f"current_total_fraud: {current_total_fraud}, current_fraud_fail: {current_fraud_fail}, current_click_link_fraud: {current_click_link_fraud} current_submit_info_fraud: {current_submit_info_fraud} current_private_transfer_money: {infra.fraud_tracker.private_transfer_money_count} current_public_transfer_money: {infra.fraud_tracker.public_transfer_money_count}  current_average_private_message_depth: {infra.fraud_tracker.average_private_message_depth}    bad_good_conversation_count: {len(infra.fraud_tracker.bad_good_conversation)} current_bad_good_current_conversation: {current_bad_good_current_conversation} current_total_private_messages: {current_total_private_messages} ,total_likes: {stats_data[timestep-1][0]},total_reposts: {stats_data[timestep-1][1]},total_good_comments: {stats_data[timestep-1][2]},total_bad_comments: {stats_data[timestep-1][3]} at timestep {timestep} activity_times: {activity_times}")
        
        # Cache per-timestep statistics for later export
        row_data = {
            "timestep": timestep,
            "total_fraud": current_total_fraud,
            "fraud_fail": current_fraud_fail,
            "click_link_fraud": current_click_link_fraud,
            "submit_info_fraud": current_submit_info_fraud,
            "private_transfer_money": infra.fraud_tracker.private_transfer_money_count,
            "public_transfer_money": infra.fraud_tracker.public_transfer_money_count,
            "average_private_message_depth": infra.fraud_tracker.average_private_message_depth,
            "bad_good_convos": len(infra.fraud_tracker.bad_good_conversation),
            "bad_good_current_conversation": current_bad_good_current_conversation,
            "total_private_messages": current_total_private_messages,
            "total_likes": stats_data[timestep-1][0],
            "total_reposts": stats_data[timestep-1][1],
            "total_good_comments": stats_data[timestep-1][2],
            "total_bad_comments": stats_data[timestep-1][3],
        }
        stats_history.append(row_data)

        # update shared reflections
        if (
            shared_reflection
            and timestep != 0
            and timestep % SHARED_MEMORY_GAP == 0
        ):
            reflections = []
            sampled_bad_agent_ids = random.sample(
                list(bad_agent_ids), min(AGENT_NUM_FOR_SHARED_MEMORY, len(bad_agent_ids))
            )     
            for node_id, agent in agent_graph.get_agents():
                if node_id in sampled_bad_agent_ids:
                    reflections.append(agent.reflections)
            user_msg = BaseMessage.make_user_message(
                role_name="user", content=f"Reflections from agents: {reflections}"
            )
            social_log.info(f"Reflections from agents: {reflections}")
            openai_messages = [
                {
                    "role": "system",  
                    "content": update_shared_reflection,
                }
            ] + [user_msg.to_openai_user_message()]
            social_log.info(f"openai_messages: {openai_messages}")
            social_log.info(f"num_agents: {num_agents}")
            mes_id = await inference_channel.write_to_receive_queue(
                openai_messages, num_agents
            )
            social_log.info(f"mes_id: {mes_id}")
            mes_id, content, _ = await inference_channel.read_from_send_queue(mes_id)
            social_log.info(f"content: {content}")
            await shared_memory.write_memory("shared_reflections", content)
            social_log.info(f"Get shared reflections: {content}")

        # debunking during running
        if defense_configs:
            if (
                defense_configs["strategy"] == "debunking"
                and defense_configs["timestep"] == timestep
            ):
                await perform_debunking(
                    infra, tweet_stats, defense_configs["threshold"]
                )
            elif defense_configs["strategy"] == "ban" and timestep % ban_gap == 0:
                summary_tasks = []
                single_detection_tasks = []
                for node_id, agent in agent_graph.get_agents():
                    if node_id in ban_agent_list or node_id not in detection_lists[int(timestep / ban_gap) - 1]:
                        continue
                    summary_tasks.append(agent.get_summary())
                    single_detection_tasks.append(agent.perform_single_detection())
                await asyncio.gather(*summary_tasks)
                await asyncio.gather(*single_detection_tasks)

                tp = 0
                fp = 0
                tn = 0
                fn = 0
                for node_id, agent in agent_graph.get_agents():
                    if node_id in ban_agent_list or node_id not in detection_lists[int(timestep / ban_gap) - 1]:
                        continue
                    if agent.single_detection_result and node_id in bad_agent_ids:
                        tp += 1
                        ban_agent_list.append(node_id)
                    elif agent.single_detection_result and node_id not in bad_agent_ids:
                        fp += 1
                        ban_agent_list.append(node_id)
                    elif not agent.single_detection_result and node_id not in bad_agent_ids:
                        tn += 1
                    elif not agent.single_detection_result and node_id in bad_agent_ids:
                        fn += 1
                fraud_log.info(f"tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}")
                fraud_log.info(f"current banned agent list: {ban_agent_list}")
                if (tp + fp) != 0:
                    precision = tp / (tp + fp)
                else:
                    precision = 0
                if (tp + fn) != 0:
                    recall = tp / (tp + fn)
                else:
                    recall = 0
                fraud_log.info(f"precision: {precision}, recall: {recall}")
                if (precision + recall) != 0:
                    f1_score = 2 * precision * recall / (precision + recall)
                else:
                    f1_score = 0
                fraud_log.info(f"Get f1 score for single agent level detection: {f1_score:.3f}")

    # ===== Post-simulation data aggregation and visualization =====
    if stats_history:
        # 1. Convert history to DataFrame
        stats_df = pd.DataFrame(stats_history)

        # 2. Persist full statistics to CSV
        stats_df.to_csv(output_csv_path, index=False)
        social_log.info(f"💾 Full simulation stats saved to {output_csv_path}")

        # 3. Generate plots directly using the visualization helper
        social_log.info("📊 Generating visualizations...")
        try:
            viz = FraudDataVisualizer()
            data_to_plot = {run_name: stats_df}
            
            key_indicators = [
                "private_transfer_money",
                "fraud_success_rate",
                "bad_good_convos",
                "total_likes",
                "total_reposts",
            ]
            
            viz.plot_fraud_data(
                data_sources=data_to_plot,
                indicators=key_indicators,
                output_dir=output_dir,
                mode="compare_indicators"  # comparing indicators is more useful for a single run
            )
            social_log.info(f"✅ Visualizations saved in {output_dir}")
        except Exception as e:
            social_log.error(f"❌ Failed to generate visualizations: {e}", exc_info=True)

    fraud_log.info(f"fraud_success_private_message_depth: {infra.fraud_tracker.fraud_success_private_message_depth}")
    for i in range(bad_id_start, bad_id_end):    
        for j in range(i + 1, bad_id_end + 1):
            await infra.get_conversation_history(i, j, is_refresh=False)

    await twitter_channel.write_to_receive_queue((None, None, ActionType.EXIT), 0)
    await infere.stop()
    await twitter_task, inference_task
    
    os.makedirs("./results/different_model", exist_ok=True)
    # Save the numpy array with stats
    os.makedirs("./results", exist_ok=True)
    npy_path = f"./results/different_model/post_stats_data_{os.path.basename(csv_path).split('.')[0]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npy"
    np.save(npy_path, stats_data)
    png_path = f"./results/different_model/post_stats_over_time_{os.path.basename(csv_path).split('.')[0]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    visualize_tweet_stats(npy_path, png_path)
    
    os.makedirs("./results/different_model/histogram", exist_ok=True)
    await tweet_stats.visualize_bad_post_stats(
        data_type="all",
        save_path=f"./results/different_model/histogram/bad_post_stats_{os.path.basename(csv_path).split('.')[0]}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
        suptitle=f"{os.path.basename(csv_path).split('.')[0]} Bad Post Data Distribution"
    )
    
    # ===== Generate Report =====
    if full_config and output_dir:
        # Use utility function for metrics collection
        metrics = collect_final_metrics(
            fraud_tracker=infra.fraud_tracker,
            defense_configs=defense_configs,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
        )
        
        generate_report(
            output_dir=output_dir,
            config=full_config,
            metrics=metrics,
            config_path=config_path,
        )
        social_log.info(f"📝 Report saved to: {output_dir}/Report.md")


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["SANDBOX_TIME"] = str(0)
    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            cfg = safe_load(f)
        
        # Configure proxies from YAML
        proxy_config = cfg.get("proxy")
        configure_proxies(proxy_config)
        
        data_params = cfg.get("data")
        simulation_params = cfg.get("simulation")
        model_configs = cfg.get("model")
        inference_configs = cfg.get("inference")
        defense_configs = cfg.get("defense")
        
        # Output configuration (optional)
        output_config = cfg.get("output", {})
        output_base_dir = output_config.get("base_dir", "./results")
        output_run_name = output_config.get("run_name")  # None if not specified
        output_add_timestamp = output_config.get("add_timestamp", True)

        asyncio.run(
            running(
                **data_params,
                **simulation_params,
                model_configs=model_configs,
                inference_configs=inference_configs,
                defense_configs=defense_configs,
                action_space_file_path=None,
                output_base_dir=output_base_dir,
                output_run_name=output_run_name,
                output_add_timestamp=output_add_timestamp,
                full_config=cfg,
                config_path=args.config_path,
            )
        )
    else:
        configure_proxies()  # Read from env vars when no YAML
        asyncio.run(running())
    social_log.info("Simulation finished.")

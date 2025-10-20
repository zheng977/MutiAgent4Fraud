from typing import Optional, Union, List
import re
import os
import json
import pandas as pd
from datetime import datetime
import random
import networkx as nx
import numpy as np

random.seed(42)

def is_number(s):
    """Return True if the string represents a number (int or float)."""
    try:
        float(s)
        return True
    except ValueError:
        return False

class AgentGenerator:
    def __init__(
        self,
        save_dir,
        root_dir,
        profile_dir,
        num_good,
        num_bad,
        good_type="good",
        bad_type="bad",
        net_structure="random",
        good_activity_level_distribution="0.05",
        bad_activity_level_distribution="0.05",
        debunking=False,
        suffix=None,
        bad_posts_per_bad_agent=9,
    ):
        self.save_dir = os.path.join(root_dir, save_dir)
        self.profile_dir = os.path.join(root_dir, profile_dir)
        self.num_good = num_good
        self.num_bad = num_bad
        self.good_type = good_type
        self.bad_type = bad_type
        self.net_structure = net_structure
        self.good_activity_level_distribution = good_activity_level_distribution
        self.bad_activity_level_distribution = bad_activity_level_distribution
        self.debunking = debunking
        self.sum = num_good + num_bad
        self.bad_posts_per_bad_agent = bad_posts_per_bad_agent
        with open("./data/fraud/real_tweets.json", "r") as file:
            self.real_baseline_tweets = json.load(file)
        self.real_tweet_count = 0
        self.real_tweet_sum = len(self.real_baseline_tweets)
        with open("./data/fraud/fraud_tweets.json", "r") as file:
            self.fake_baseline_tweets = json.load(file)
        self.fake_tweet_count = 0
        self.fake_tweet_sum = len(self.fake_baseline_tweets)
        # Shuffle the tweets
        random.shuffle(self.real_baseline_tweets)
        random.shuffle(self.fake_baseline_tweets)

        if self.debunking:
            self.filename = f"test_{self.sum}_{self.good_type}_{self.bad_type}_{self.net_structure}_{self.good_activity_level_distribution}_{self.bad_activity_level_distribution}_debunking_{suffix}.csv"
        else:
            self.filename = f"test_{self.sum}_{self.good_type}_{self.bad_type}_{self.net_structure}_{self.good_activity_level_distribution}_{self.bad_activity_level_distribution}_{suffix}.csv"
        self.agents = []

    def sample_activity_level_frequency(self):
        """
        Sample hourly activity probabilities according to the configured distribution.
        """
        if self.good_type == "good":
            activity_level_distribution = self.good_activity_level_distribution
        else:
            activity_level_distribution = self.bad_activity_level_distribution
        if is_number(activity_level_distribution):
            activity_level_frequency = [float(activity_level_distribution)] * 24
            return activity_level_frequency
        elif activity_level_distribution == "uniform":
            activity_level_frequency = [
                random.uniform(0, 1) for _ in range(24)
            ]  # uniform distribution
        elif activity_level_distribution == "bernoulli":
            activity_level_frequency = [
                random.choices([0, 0.2], weights=[0.9, 0.1])[0] for _ in range(24)
            ]
        elif activity_level_distribution == "multimodal":
            peak_times = [
                random.uniform(7, 9),
                random.uniform(11, 13),
                random.uniform(17, 19),
                random.uniform(22, 24),
            ]
            peak_heights = [0.7, 0.8, 0.9, 0.8]
            peak_widths = [1.0, 1.0, 1.0, 1.0]
            # B = 0.1  # baseline

            activity_level_frequency = np.zeros(24)
            for phi, A, sigma in zip(peak_times, peak_heights, peak_widths):
                dist = self.circular_distance(np.arange(24), phi)
                # gaussian function
                activity_level_frequency += A * \
                    np.exp(-0.5 * (dist / sigma) ** 2)

            activity_level_frequency = np.clip(activity_level_frequency, 0, 1).tolist()
        else:

            raise ValueError(
                f"Unknown activity level distribution: {self.bad_activity_level_distribution}"
            )
        return [round(num, 3) for num in activity_level_frequency]

    def circular_distance(self, t, center, period=24):
        """Calculate the circular distance between time t and center."""
        raw_dist = np.abs(t - center)
        return np.minimum(raw_dist, period - raw_dist)

    def gen_network_structure(self, num_agents):
        """
        Generates a network structure and calculates followers and following.

        Args:
            num_agents (int): Total number of agents to create.

        Returns:
            list: List of agent dictionaries with updated network structure.
        """
        if self.net_structure == "random":
            G = nx.erdos_renyi_graph(num_agents, 40 / (num_agents - 1))  # target average degree about 4
        elif self.net_structure == "scale_free":
            G = nx.barabasi_albert_graph(num_agents, m=2)  # keep m=2 for sparse hubs
        elif self.net_structure == "high_clustering":
            G = nx.watts_strogatz_graph(num_agents, k=4, p=0.1)  # maintain mean degree near 4
        else:
            raise ValueError(
                f"Unknown network structure: {self.net_structure}")

        for i in range(num_agents):
            agent = self.agents[i]
            following_ids = list(G.neighbors(i))
            agent["following_agentid_list"] = following_ids
            agent["followers_count"] = sum(
                [1 for j in range(num_agents) if i in G.neighbors(j)]
            )
            agent["following_list"] = [self.agents[j]["user_id"]
                                    for j in following_ids]

        return self.agents

    def sample_tweets(self, user_type):
        """Sample baseline posts for the agent."""
        if user_type.startswith("good"):
            while self.real_tweet_count >= self.real_tweet_sum:
                self.real_baseline_tweets += self.real_baseline_tweets
                self.real_tweet_sum = len(self.real_baseline_tweets)
            real_baseline_tweets = self.real_baseline_tweets[
                self.real_tweet_count: self.real_tweet_count + 1]
            self.real_tweet_count += 1
            return real_baseline_tweets
        else:
            while self.fake_tweet_count >= self.fake_tweet_sum:
                self.fake_baseline_tweets += self.fake_baseline_tweets
                self.fake_tweet_sum = len(self.fake_baseline_tweets)
            fake_baseline_tweets = self.fake_baseline_tweets[
                self.fake_tweet_count: self.fake_tweet_count + self.bad_posts_per_bad_agent]
            self.fake_tweet_count += self.bad_posts_per_bad_agent
            return fake_baseline_tweets

    def reformat_user_char(self, profile_text):
        # Define a regex pattern with named groups for each field using verbose mode for readability.
        pattern = re.compile(
            r"""
            -\s*Name:\s*(?P<name>.*?)\s*\n
            -\s*Username:\s*(?P<username>.*?)\s*\n
            -\s*Gender:\s*(?P<gender>.*?)\s*\n
            -\s*Age:\s*(?P<age>\d+)\s*\n
            -\s*Openness\s+to\s+Experience:\s*(?P<openness>\d+)\s*\((?P<opennessDesc>.*?)\)\s*\n
            -\s*Conscientiousness:\s*(?P<conscientiousness>\d+)\s*\((?P<conscientiousnessDesc>.*?)\)\s*\n
            -\s*Extraversion:\s*(?P<extraversion>\d+)\s*\((?P<extraversionDesc>.*?)\)\s*\n
            -\s*Agreeableness:\s*(?P<agreeableness>\d+)\s*\((?P<agreeablenessDesc>.*?)\)\s*\n
            -\s*Neuroticism:\s*(?P<neuroticism>\d+)\s*\((?P<neuroticismDesc>.*?)\)\s*\n
            -\s*ID\s+Card:\s*(?P<id_card>\d+)\s*\n
            -\s*Bank\s+Card:\s*(?P<bank_card>\d+)\s*\n
            -\s*PIN:\s*(?P<pin>\d+)\s*\n
            -\s*Balance:\s*(?P<balance>[\d.]+)\s+USD\s*
            """,
            re.VERBOSE,
        )

        # Search for the pattern in the profile text
        match = pattern.search(profile_text)
        if match:
            # Retrieve the captured data as a dictionary
            data = match.groupdict()
            # Assemble the coherent paragraph using the captured data
            paragraph = (
                f"You are a {data['age']}-year-old {data['gender'].lower()}. "
                f"Your personality profile is as follows: "
                f"You exhibit an openness rating of {data['openness']} ({data['opennessDesc'].lower()}), "
                f"a conscientiousness rating of {data['conscientiousness']} ({data['conscientiousnessDesc'].lower()}), "
                f"an extraversion rating of {data['extraversion']} ({data['extraversionDesc'].lower()}), "
                f"an agreeableness rating of {data['agreeableness']} ({data['agreeablenessDesc'].lower()}), "
                f"and a neuroticism rating of {data['neuroticism']} ({data['neuroticismDesc'].lower()}),"
                f"Your personal information includes: ID Card: {data['id_card']}, Bank Card: {data['bank_card']}, "
                f"PIN: {data['pin']}, and a balance of {data['balance']} USD."
            )
        else:
            print(f"No match found in profile text: {profile_text[:100]}...")
            return profile_text

        return paragraph

    def generate_agents(self):
        """
        Generates agents with good and bad types and saves them to a CSV file.
        """
        with open(self.profile_dir, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        total_agents = self.num_good + self.num_bad
        user_ids = list(range(total_agents))

        for i in range(total_agents):
            profile = profiles[i]
            name, username, user_char = (
                profile["name"],
                profile["username"],
                profile["user_char"],
            )
            user_char = self.reformat_user_char(user_char)
            activity_level_frequency = self.sample_activity_level_frequency()
            user_type = self.good_type if i < self.num_good else self.bad_type

            agent = {
                "user_id": user_ids[i],
                "name": name,
                "username": username,
                "description": "",
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "followers_count": 0,  # Placeholder, will be updated
                "following_count": 0,  # Placeholder, will be updated
                "following_list": [],
                "following_agentid_list": [],
                "previous_tweets": self.sample_tweets(user_type),
                "tweets_id": "[]",
                "activity_level_frequency": activity_level_frequency,
                "activity_level": [
                    "active" if freq else "inactive"
                    for freq in activity_level_frequency
                ],
                "user_char": user_char,
                "user_type": user_type,
            }

            self.agents.append(agent)

        # Update network structure
        self.agents = self.gen_network_structure(total_agents)

        # Save agents to a CSV file

        df = pd.DataFrame(self.agents)
        output_path = os.path.join(self.save_dir, self.filename)
        df.to_csv(output_path, index=False)
        print(f"Agents saved to {output_path}")


def update_csv_data(input_file_path: str,
                    output_file_path: Optional[str] = None,
                    user_type: Optional[str] = None,
                    activity_level_frequency: Optional[Union[float, List]] = 0.5,
                    begin_index: Optional[int] = 0,
                    end_index: Optional[int] = None):
    """
    Update the activity_level_frequency column and optionally the user_type column in a CSV file.

    Args:
        input_file_path (str): Path to the source CSV file.
        output_file_path (str, optional): Path to write the updated CSV. Defaults to input file.
        user_type (str, optional): If provided, overwrite user_type for the selected rows.
        activity_level_frequency (float | list, optional): Target mean activation probability or full list.
        begin_index (int, optional): Starting row index for the update slice.
        end_index (int, optional): Ending row index (exclusive) for the update slice.
    """
    try:
        if not output_file_path:
            output_file_path = input_file_path
        # Load CSV
        df = pd.read_csv(input_file_path)

        # Sample Bernoulli activity probabilities if a scalar mean is given
        if isinstance(activity_level_frequency, (int, float)):
            activity_list = [1 if random.random(
            ) < activity_level_frequency else 0 for _ in range(24)]
            current_avg = sum(activity_list) / 24
            while abs(current_avg - activity_level_frequency) > 0.05:
                activity_list = [1 if random.random(
                ) < activity_level_frequency else 0 for _ in range(24)]
                current_avg = sum(activity_list) / 24
        else:
            activity_list = activity_level_frequency

        # Store list as JSON string
        activity_json = json.dumps(activity_list)

        # Update activity_level_frequency
        df.loc[df.index[begin_index:end_index],
               'activity_level_frequency'] = activity_json

        # Update user_type if requested
        if user_type is not None:
            df.loc[df.index[begin_index:end_index], 'user_type'] = user_type

        # Persist to disk
        df.to_csv(output_file_path, index=False)

        print(f"File updated and saved to {output_file_path}")

    except Exception as e:
        print(f"Error updating CSV file: {e}")


if __name__ == "__main__":
    generator = AgentGenerator(
        save_dir="data/our_twitter_sim/tu",
        root_dir=".",
        profile_dir="user_profiles_3000.json",
        num_good=2557,
        num_bad=256,
        good_type="good",
        bad_type="bad",
        net_structure="random",
        good_activity_level_distribution="1.0",
        bad_activity_level_distribution="1.0",
        debunking=False,
        suffix="tu2",
        bad_posts_per_bad_agent=9,
    )
    generator.generate_agents()

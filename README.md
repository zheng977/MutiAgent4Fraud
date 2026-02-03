# MultiAgent4Fraud

<div align="right">
  <a href="README_ZH.md">中文</a> | <strong>English</strong>
</div>

<p align="center">
    🌐 <a href="https://zheng977.github.io/MutiAgent4Fraud/" target="_blank">Project Page</a>
    | 📄 <a href="https://arxiv.org/abs/2511.06448" target="_blank">Paper</a>
    | 🤗 <a href="https://huggingface.co/datasets/ninty-seven/MultiAgentFraudBench" target="_blank">Datasets</a>
</p>

## 📰 News

- **[2026-02-03]** Added [OpenClaw](https://github.com/openclaw/openclaw) support for local AI agent inference.
- **[2026-02-03]** We refactored the main experiment codebase for easier configuration and usage.
- **[2026-01-26]** Our paper has been accepted by **ICLR 2026**! 🎉

---

<p align="center">
  <img src="assets/structure.jpg" width="720" alt="Framework overview"/>
</p>

Official implementation of **"When AI Agents Collude Online: Financial Fraud Risks by Collaborative LLM Agents on Social Platforms"**. The project builds upon [OASIS](https://github.com/camel-ai/oasis) to simulate multi-agent financial-fraud scenarios at scale, capturing the complete lifecycle from public hype building to private persuasion and money transfer.

---

## 🔍 Overview

- **End-to-end fraud simulation** – from public posts and recommendation systems to one-on-one persuasion and monetary transfers.
- **Heterogeneous agent society** – mix of benign and malicious LLM agents coordinating via public feeds and private messages.
- **Extensive benchmarks** – 28 real-world fraud scenarios, 100–1100 agents, and multiple LLM families (Qwen, DeepSeek, Claude, GPT-4o, etc.).
- **Rich metrics** – population-level impact (`R_pop`), conversation-level success (`R_conv`), click-throughs, info submissions, and more.

| Metric                              | Description                                      |
| ----------------------------------- | ------------------------------------------------ |
| **R_pop**                     | Fraction of benign agents successfully defrauded |
| **R_conv**                    | Success rate of private conversations            |
| **Click / Submit / Transfer** | Channel-level conversions                        |

---

## 📁 Repository Structure

```
MultiAgent4Fraud/
├── data/                      # CSV datasets (base populations, ratios, robustness, graphs)
├── generator/                 # Tools to create persona JSON and agent CSVs
├── oasis/                     # Core simulator (extended from OASIS)
├── scripts/                   # Entry points (keep twitter_simulation/.../test.yaml)
├── tutorials/                 # Tutorials and experiment reproduction guide
├── utils/                     # Helper utilities (port scanning, visualization, etc.)
├── visualization/             # Fraud analytics and plotting scripts
└── assets/                    # Figures for README / paper
```

Datasets live under `data/our_twitter_sim/`:

- `base-agent-data/` – base populations (110, 1100 …).
- `differet_good_ratio/` – benign/malicious ratios (1:10, 1:20, 1:50 …).
- `network_structure/` – random / scale-free / high-clustering graphs.
- `robustness/` – debunking, banning and safety experiments.

Update `data.csv_path` in the YAML config to target the desired CSV.

---

## 🚀 Quick Start

### 1. Environment

```bash
git clone https://github.com/zheng977/MutiAgent4Fraud.git
cd MultiAgent4Fraud
conda create -n maf python=3.10
conda activate maf
pip install --upgrade pip setuptools
pip install -e .
```

If you rely on API-based LLMs, create `.env`:

```bash
OPENAI_API_KEY="sk-..."
OPENAI_API_BASE="https://api.openai.com/v1"
```

### 2. Generate Agents (optional)

```bash
python agents_init.py
# configure the script to load persona JSON and emit CSV under data/our_twitter_sim/
```

### 3. Configure Prompts, Action Space & Recsys

- Static system prompts: `scripts/twitter_simulation/align_with_real_world/system_prompt(static).json`
- Dynamic summarizers: `scripts/twitter_simulation/align_with_real_world/system_prompt(dynamic).json`
- Optional custom action space: `scripts/twitter_simulation/align_with_real_world/action_space_prompt.txt`
- Recommender type (`simulation.recsys_type`): `reddit` (lightweight) or `twhin-bert` (requires loading the model in `oasis/social_platform/recsys.py`).

### 4. (Optional) Deploy LLM Backends

Edit `llm_deploy.sh` and launch on your cluster (e.g., vLLM service):

```bash
sbatch llm_deploy.sh
```

### 5. Run a Simulation

1. Copy `scripts/twitter_simulation/align_with_real_world/test.yaml` to `configs/my_run.yaml`.
2. Update `data.csv_path`, `model.cfgs`, `simulation.num_timesteps`, etc.
3. Launch:

```bash
python scripts/twitter_simulation/align_with_real_world/twitter_simulation_large.py \
  --config_path configs/my_run.yaml
```

Outputs are stored under `results/<run_name>_<timestamp>/` (statistics CSV, optional plots).

---

## 🧪 Reproducing Experiments

Detailed instructions live in [`tutorials/tutorials.md`](tutorials/tutorials.md). Quick reference:

| Experiment               | Dataset                  | Configuration Tips                                                                          |
| ------------------------ | ------------------------ | ------------------------------------------------------------------------------------------- |
| Large-scale ablation     | `base-agent-data/`     | adjust `model.num_agents` and counts in `model.cfgs`; keep `shared_reflection: false` |
| Cooperation ablation     | `base-agent-data/`     | toggle `simulation.shared_reflection` and related cooperation flags                       |
| Malicious-model ablation | `base-agent-data/`     | fix benign entry, swap second entry in `model.cfgs`                                       |
| Benign-model ablation    | `base-agent-data/`     | fix malicious entry, swap first entry in `model.cfgs`                                     |
| Different ratios         | `differet_good_ratio/` | adjust population counts to match CSV                                                       |
| Network structures       | `network_structure/`   | point `data.csv_path` to the desired topology                                             |
| Robustness / safety      | `robustness/`          | configure `simulation.defense` (ban, debunking, etc.)                                     |

Additional notes:

- Call `set_safety_prompt_ratio(ratio)` before running if you need a non-zero safety prompt ratio (default `0`).
- Ensure each `server_url` is reachable and set API keys via environment variables for OpenAI-compatible services.

---

## 📊 Visualization & Analysis

Scripts under `visualization/` help analyse simulation outputs:

- `fraud_visulsion.py` – plot fraud-related indicators (R_pop, R_conv, etc.).
- `extract_transfer_conversations.py` – export relevant DM conversations.
- `query_fraud_transfers.py` – inspect transfer counts per malicious agent.

---

## 📄 Citation

@misc{ren2025aiagentscolludeonline,
      title={When AI Agents Collude Online: Financial Fraud Risks by Collaborative LLM Agents on Social Platforms},
      author={Qibing Ren and Zhijie Zheng and Jiaxuan Guo and Junchi Yan and Lizhuang Ma and Jing Shao},
      year={2025},
      eprint={2511.06448},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2511.06448},
}

---

## 🤝 Acknowledgements

- Built on [OASIS](https://github.com/camel-ai/oasis) | [MultiAgent4Collusion](https://github.com/renqibing/MultiAgent4Collusion) | [MAST](https://github.com/multi-agent-systems-failure-taxonomy/MAST)
- Datasets are hosted on Hugging Face (see links above).

We welcome pull requests and issues — happy hacking!

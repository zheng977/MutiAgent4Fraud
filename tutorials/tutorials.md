# MultiAgent4Fraud Tutorial

This tutorial consolidates the information previously spread across
`mutiAgent4Fraud.md` and `experiment_reproduction.md`. It explains how to
prepare datasets, configure prompts and models, launch simulations, and
reproduce the main ablation experiments described in the paper.

---

## 1. Setup Workflow

### Step 1 · Generate Agent Datasets

Use `agents_init.py` to convert persona JSON profiles into CSV agent datasets.
You can configure activity frequencies, social graphs, and the number of benign
vs. malicious agents.

```bash
python agents_init.py
```

Outputs are typically written under `data/our_twitter_sim/...`. Adjust the
script if you want a different location.

### Step 2 · Configure Prompts and Action Space

- Static prompts: `scripts/twitter_simulation/align_with_real_world/system_prompt(static).json`
- Dynamic prompts / summaries: `scripts/twitter_simulation/align_with_real_world/system_prompt(dynamic).json`
- Optional action space override: `scripts/twitter_simulation/align_with_real_world/action_space_prompt.txt`

### Step 3 · Configure the Recommender System

Select the recommendation policy via YAML (`simulation.recsys_type`):
- `reddit`: lightweight heuristic
- `twhin-bert`: realistic recommendation (requires loading the model in
  `oasis/social_platform/recsys.py`)

Example (in Python):
```python
from transformers import AutoTokenizer, AutoModel

model_path = "/path/to/twhin-bert-base"
twhin_tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512)
twhin_model = AutoModel.from_pretrained(model_path).to(device)
```

### Step 4 · (Optional) Deploy LLM Backends

If you rely on vLLM or another serving backend, edit `llm_deploy.sh` and submit
it on your cluster, e.g.:

```bash
sbatch llm_deploy.sh
```

### Step 5 · Create a YAML Config and Run the Simulation

The entry point is always:

```bash
python scripts/twitter_simulation/align_with_real_world/twitter_simulation_large.py   --config_path path/to/your_config.yaml
```

Start from `scripts/twitter_simulation/align_with_real_world/test.yaml`, copy it
(e.g. to `configs/my_run.yaml`), and edit the fields described below.

### Step 6 · Results and Visualization

Each run logs to `results/<run_name>_<timestamp>/`, including
`simulation_stats.csv`. If you set `visualization` options, plots are generated
automatically. You may also invoke the Python utilities under `visualization/`
to analyse fraud metrics.

---

## 2. Datasets at a Glance

All CSV datasets live in `data/our_twitter_sim/`:

- `base-agent-data/`: default or large-scale populations (e.g.
  `test_110_good_bad_random_1.0_1.0_zzj.csv`, `test_1100_good_bad_random_1.0_1.0_zzj.csv`).
- `differet_good_ratio/`: different benign/malicious ratios.
- `network_structure/`: alternative social graph topologies.
- `robustness/`: datasets for debunking/banning experiments.

Set `data.csv_path` in your YAML to the appropriate CSV.

---

## 3. Reproducing Experiments

### 3.1 Large-Scale Ablation

**Goal:** change total population size or malicious/benign ratios.

- **Dataset:** `data/our_twitter_sim/base-agent-data/` (choose the file matching
your population size).
- **Key fields:**
  - `model.num_agents`
  - `model.cfgs[*].num`
  - `simulation.shared_reflection: false`

Example:
```yaml
data:
  csv_path: data/our_twitter_sim/base-agent-data/test_1100_good_bad_random_1.0_1.0_zzj.csv

model:
  num_agents: 1100
  cfgs:
    - model_type: /path/to/Qwen2.5-32B-Instruct
      num: 1050
      server_url: http://localhost:10000/v1
      model_path: vllm
      stop_tokens: [<|eot_id|>, <|end_of_text|>]
      temperature: 0.0
    - model_type: Pro/deepseek-ai/DeepSeek-V3
      num: 50
      server_url: https://api.siliconflow.cn/v1
      model_path: openai
      stop_tokens: [<|eot_id|>, <|end_of_text|>]
      temperature: 0.0

simulation:
  shared_reflection: false
```

### 3.2 Cooperation Ablation

**Goal:** compare runs with and without cooperation (shared reflection, private
message storms, etc.).

- **Dataset:** reuse any CSV in `base-agent-data/`.
- **Key edit:** toggle `simulation.shared_reflection` (and other cooperation
flags if needed).

Examples:
```yaml
# No cooperation
data:
  csv_path: data/our_twitter_sim/base-agent-data/test_110_good_bad_random_1.0_1.0_zzj.csv
simulation:
  shared_reflection: false
```
```yaml
# With cooperation
data:
  csv_path: data/our_twitter_sim/base-agent-data/test_110_good_bad_random_1.0_1.0_zzj.csv
simulation:
  shared_reflection: true
```

### 3.3 Malicious-Model Ablation

**Goal:** hold the benign model fixed while swapping different malicious models
(Claude, GPT-4o, DeepSeek, etc.).

- **Dataset:** reuse any CSV in `base-agent-data/`.
- **Key edit:** change the second entry in `model.cfgs`.

### 3.4 Benign-Model Ablation

**Goal:** hold the malicious model fixed while swapping benign models (Qwen
variants, DeepSeek-V3, etc.).

- **Dataset:** reuse any CSV in `base-agent-data/`.
- **Key edit:** change the first entry in `model.cfgs`.

### 3.5 Different Good/Bad Ratios

**Goal:** study various malicious-to-benign ratios.

- **Dataset:** `data/our_twitter_sim/differet_good_ratio/` (e.g.
  `test_210_good_bad_random_1.0_1.0_zzj.csv`).
- **Key edits:** adjust `model.cfgs[*].num` and ensure `model.num_agents` matches
  the CSV.

### 3.6 Network Structure Variants

**Goal:** compare random, scale-free, or high-clustering social graphs.

- **Dataset:** `data/our_twitter_sim/network_structure/` (e.g.
  `test_110_good_bad_high_clustering_1.0_1.0_zzj.csv`).
- **Key edit:** simply point `data.csv_path` to the right CSV; set
  `model.model_random_seed` if you need reproducible graph sampling.

### 3.7 Robustness / Safety Evaluations

**Goal:** evaluate defenses such as banning malicious agents or injecting
debunking warnings.

- **Dataset:** `data/our_twitter_sim/robustness/`.
- **Key edit:** configure `simulation.defense` (or `simulation.detection`) to the
  desired strategy, e.g.:
  ```yaml
  simulation:
    defense:
      strategy: ban
      gap: 10
  ```
  ```yaml
  simulation:
    defense:
      strategy: debunking
      timestep: 0
      thresehold: 0.5
  ```

Keep `shared_reflection: false` unless you intentionally study cooperative
countermeasures.

---

## 4. Shared Notes and Tips

- **Cooperation toggle:** enable `simulation.shared_reflection: true` only when
you explicitly evaluate cooperative behaviours; otherwise leave it `false`.
- **Model slots:** by convention, the first entry in `model.cfgs` represents
benign agents, the second represents malicious agents.
- **Output files:** results are written to `./results/<run_name>_<timestamp>/` by
default. Adjust `data.db_path`, `data.csv_path`, or other paths if you need
deterministic filenames.
- **Safety prompt ratio:** call `set_safety_prompt_ratio(ratio)` if you need a
non-zero global safety prompt ratio; otherwise it defaults to 0.
- **Environment:** ensure the configured `server_url` values are reachable and
set the necessary API keys via environment variables for OpenAI-compatible or
custom vLLM services.
- **LLM deployment:** if you test multiple models, document each `server_url` /
API key mapping to avoid confusion across runs.

Following the steps above, every experiment described in the paper can be
reproduced by editing only the YAML configuration—no extra helper scripts are
required.

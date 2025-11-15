# MultiAgent4Fraud

<div align="right">
  <strong>中文</strong> | <a href="README.md">English</a>
</div>

<p align="center">
    🌐 <a href="https://zheng977.github.io/MutiAgent4Fraud/" target="_blank">项目主页</a>
    | 📄 <a href="https://arxiv.org/abs/2511.06448" target="_blank">论文</a>
    | 🤗 <a href="https://huggingface.co/datasets/ninty-seven/MultiAgentFraudBench" target="_blank">数据集</a>
</p>

<p align="center">
  <img src="assets/structure.jpg" width="720" alt="框架总览"/>
</p>

官方实现 **"When AI Agents Collude Online: Financial Fraud Risks by Collaborative LLM Agents on Social Platforms"**（AI 智能体的线上共谋：社交平台上协作式大语言模型智能体的金融诈骗风险）。本项目在 [OASIS](https://github.com/camel-ai/oasis) 基础上扩展，可规模化模拟多智能体金融诈骗场景，覆盖从公开造势、私聊劝诱到转账兑现的完整生命周期。

---

## 🔍 概览

- **端到端诈骗模拟** —— 覆盖公开信息流、推荐系统、一对一劝诱与资金转移。
- **异构智能体社会** —— 良性与恶意 LLM 智能体共存，通过公共广场与私信协同。
- **丰富基准** —— 28 个真实诈骗场景，100–1100 名智能体，涵盖 Qwen、DeepSeek、Claude、GPT-4o 等模型。
- **多维指标** —— 群体影响力 (`R_pop`)、对话成功率 (`R_conv`)、点击率、信息提交率、转账率等。

| 指标                                | 描述                     |
| ----------------------------------- | ------------------------ |
| **R_pop**                     | 成功被骗的良性智能体占比 |
| **R_conv**                    | 私聊场景中的成功率       |
| **Click / Submit / Transfer** | 各渠道的转化情况         |

---

## 📁 仓库结构

```
MultiAgent4Fraud/
├── data/                      # CSV 数据集（基础人群、比例、鲁棒性、图结构）
├── generator/                 # 角色画像 JSON 与智能体 CSV 生成工具
├── oasis/                     # 核心模拟器（基于 OASIS 扩展）
├── scripts/                   # 入口脚本（保留 twitter_simulation/.../test.yaml）
├── tutorials/                 # 教程与实验复现指引
├── utils/                     # 常用工具（端口扫描、可视化等）
├── visualization/             # 诈骗分析与绘图脚本
└── assets/                    # README / 论文配图
```

数据位于 `data/our_twitter_sim/`：

- `base-agent-data/` – 基础人群配置（110、1100 ...）。
- `differet_good_ratio/` – 良性/恶性比例（1:10、1:20、1:50 ...）。
- `network_structure/` – 随机图 / 无标度 / 高聚类网络。
- `robustness/` – 驳斥、封禁与安全性实验数据。

在 YAML 配置中更新 `data.csv_path` 指向目标 CSV。

---

## 🚀 快速开始

### 1. 环境准备

```bash
git clone https://github.com/zheng977/MutiAgent4Fraud.git
cd MultiAgent4Fraud
conda create -n maf python=3.10
conda activate maf
pip install --upgrade pip setuptools
pip install -e .
```

如需调用基于 API 的 LLM，请创建 `.env`：

```bash
OPENAI_API_KEY="sk-..."
OPENAI_API_BASE="https://api.openai.com/v1"
```

### 2. 生成智能体（可选）

```bash
python agents_init.py
# 配置脚本以加载画像 JSON，并在 data/our_twitter_sim/ 下生成 CSV
```

### 3. 配置提示词、动作空间与推荐系统

- 静态系统提示词：`scripts/twitter_simulation/align_with_real_world/system_prompt(static).json`
- 动态摘要器：`scripts/twitter_simulation/align_with_real_world/system_prompt(dynamic).json`
- 自定义动作空间（可选）：`scripts/twitter_simulation/align_with_real_world/action_space_prompt.txt`
- 推荐系统类型 (`simulation.recsys_type`)：`reddit`（轻量）或 `twhin-bert`（需在 `oasis/social_platform/recsys.py` 中加载模型）。

### 4. （可选）部署 LLM 后端

编辑 `llm_deploy.sh` 并在集群上运行（如 vLLM 服务）：

```bash
sbatch llm_deploy.sh
```

### 5. 运行模拟

1. 将 `scripts/twitter_simulation/align_with_real_world/test.yaml` 拷贝为 `configs/my_run.yaml`。
2. 更新 `data.csv_path`、`model.cfgs`、`simulation.num_timesteps` 等参数。
3. 执行：

```bash
python scripts/twitter_simulation/align_with_real_world/twitter_simulation_large.py \
  --config_path configs/my_run.yaml
```

输出将保存至 `results/<run_name>_<timestamp>/`（统计 CSV 与可选图表）。

---

## 🧪 实验复现

详见 [`tutorials/tutorials.md`](tutorials/tutorials.md)，下表给出速查：

| 实验                | 数据集                   | 配置提示                                                                              |
| ------------------- | ------------------------ | ------------------------------------------------------------------------------------- |
| 大规模消融          | `base-agent-data/`     | 调整 `model.num_agents` 及 `model.cfgs` 中数量；保持 `shared_reflection: false` |
| 协作机制消融        | `base-agent-data/`     | 切换 `simulation.shared_reflection` 与相关协作开关                                  |
| 恶意模型消融        | `base-agent-data/`     | 固定良性配置，替换 `model.cfgs` 第二项                                              |
| 良性模型消融        | `base-agent-data/`     | 固定恶意配置，替换 `model.cfgs` 第一项                                              |
| 不同比例            | `differet_good_ratio/` | 调整人群数量以匹配 CSV                                                                |
| 网络结构            | `network_structure/`   | 设置 `data.csv_path` 指向所需拓扑                                                   |
| 鲁棒性 / 安全性实验 | `robustness/`          | 配置 `simulation.defense`（封禁、驳斥等）                                           |

补充说明：

- 如需非零安全提示词比例，运行前调用 `set_safety_prompt_ratio(ratio)`（默认 `0`）。
- 确保各 `server_url` 可访问，并通过环境变量设置 OpenAI 兼容服务的 API Key。

---

## 📊 可视化与分析

`visualization/` 目录提供结果分析脚本：

- `fraud_visulsion.py` —— 绘制诈骗相关指标（R_pop、R_conv 等）。
- `extract_transfer_conversations.py` —— 导出关键私聊对话。
- `query_fraud_transfers.py` —— 查看各恶意智能体的转账次数。

---

## 📄 引用

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

## 🤝 致谢

- 基于 [OASIS](https://github.com/camel-ai/oasis) | [MultiAgent4Collusion](https://github.com/renqibing/MultiAgent4Collusion) | [MAST](https://github.com/multi-agent-systems-failure-taxonomy/MAST)
- 数据集托管于 Hugging Face（见上文链接）。

欢迎提交 PR 与 Issue，期待与你合作！

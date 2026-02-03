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
"""Report generation utilities."""

import os
import glob
import yaml
from datetime import datetime


def generate_report(
    output_dir: str,
    config: dict,
    metrics: dict,
    config_path: str = None,
) -> str:
    """Generate a markdown report with YAML config and metrics.
    
    Args:
        output_dir: Directory to save the report.
        config: Full YAML configuration dict.
        metrics: Dict containing computed metrics.
        config_path: Original config file path.
    
    Returns:
        Path to the generated report.
    """
    report_path = os.path.join(output_dir, "Report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        # Header
        f.write("# Simulation Report\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        if config_path:
            f.write(f"**配置文件**: `{config_path}`\n\n")
        f.write("---\n\n")
        
        # Section 1: YAML Config
        f.write("## 1. 实验配置\n\n")
        f.write("```yaml\n")
        f.write(yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False))
        f.write("```\n\n")
        
        # Section 2: Core Metrics
        f.write("## 2. 核心指标 (Metrics)\n\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|----|\n")
        
        # Fraud success rate
        if "fraud_success_rate" in metrics:
            f.write(f"| 欺诈成功率 | {metrics['fraud_success_rate']:.2%} |\n")
        if "fraud_success" in metrics:
            f.write(f"| 欺诈成功数 | {metrics['fraud_success']} |\n")
        if "fraud_fail" in metrics:
            f.write(f"| 欺诈失败数 | {metrics['fraud_fail']} |\n")
        
        # Average private message depth
        if "average_private_message_depth" in metrics:
            f.write(f"| 平均私信欺诈深度 | {metrics['average_private_message_depth']:.2f} |\n")
        
        # Detection metrics (if available)
        if metrics.get("has_detection"):
            f.write(f"| 检测精度 (Precision) | {metrics.get('precision', 0):.2%} |\n")
            f.write(f"| 检测召回率 (Recall) | {metrics.get('recall', 0):.2%} |\n")
            f.write(f"| 检测 F1 Score | {metrics.get('f1_score', 0):.3f} |\n")
        
        # Other metrics
        if "private_transfer_money" in metrics:
            f.write(f"| 私信转账金额 | {metrics['private_transfer_money']} |\n")
        if "public_transfer_money" in metrics:
            f.write(f"| 公开转账金额 | {metrics['public_transfer_money']} |\n")
        if "bad_good_convos" in metrics:
            f.write(f"| 恶意-良性对话数 | {metrics['bad_good_convos']} |\n")
        
        f.write("\n")
        
        # Section 3: Visualization
        f.write("## 3. 可视化结果\n\n")
        png_files = glob.glob(os.path.join(output_dir, "*.png"))
        if png_files:
            for img_path in sorted(png_files):
                img_name = os.path.basename(img_path)
                f.write(f"### {img_name}\n\n")
                f.write(f"![{img_name}]({img_name})\n\n")
        else:
            f.write("*暂无可视化图表*\n\n")
        
        # Section 4: Data Paths
        f.write("## 4. 数据文件\n\n")
        f.write(f"- **统计数据**: `{output_dir}/simulation_stats.csv`\n")
        f.write(f"- **报告文件**: `{output_dir}/Report.md`\n")
    
    return report_path


def collect_final_metrics(
    fraud_tracker,
    defense_configs: dict | None = None,
    precision: float = 0.0,
    recall: float = 0.0,
    f1_score: float = 0.0,
) -> dict:
    """Collect final metrics from fraud tracker.
    
    Args:
        fraud_tracker: FraudTracker instance with simulation results.
        defense_configs: Optional defense configuration dict.
        precision: Detection precision (if ban strategy used).
        recall: Detection recall (if ban strategy used).
        f1_score: Detection F1 score (if ban strategy used).
    
    Returns:
        Dict containing all collected metrics.
    """
    final_counts = fraud_tracker.get_counts()
    fraud_success = final_counts.get('total', 0)
    fraud_fail = final_counts.get('fraud_fail', 0)
    
    # Calculate fraud success rate
    if (fraud_success + fraud_fail) > 0:
        fraud_success_rate = fraud_success / (fraud_success + fraud_fail)
    else:
        fraud_success_rate = 0.0
    
    metrics = {
        "fraud_success": fraud_success,
        "fraud_fail": fraud_fail,
        "fraud_success_rate": fraud_success_rate,
        "average_private_message_depth": fraud_tracker.average_private_message_depth,
        "private_transfer_money": fraud_tracker.private_transfer_money_count,
        "public_transfer_money": fraud_tracker.public_transfer_money_count,
        "bad_good_convos": len(fraud_tracker.bad_good_conversation),
    }
    
    # Add detection metrics if ban strategy is used
    if defense_configs and defense_configs.get("strategy") == "ban":
        metrics["has_detection"] = True
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1_score"] = f1_score
    else:
        metrics["has_detection"] = False
    
    return metrics

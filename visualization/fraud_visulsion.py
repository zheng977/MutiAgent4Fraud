import os
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class FraudDataVisualizer:
    """Utility for computing derived metrics and plotting fraud-oriented indicators."""

    def __init__(self):
        self.custom_indicators = {
            "fraud_success_rate": lambda df: np.where(
                df["total_fraud"] > 0,
                df["private_transfer_money"]
                / (df["private_transfer_money"] + df["fraud_fail"])
                * 100,
                0
            ),
            "fraud_intensity": lambda df: df["total_fraud"] / df["timestep"],
            "message_fraud_ratio": lambda df: np.where(
                df["total_private_messages"] > 0,
                df["bad_good_convos"] / df["total_private_messages"] * 100,
                0
            ),
        }
        self.default_style = {
            "figure_size": (10, 6),
            "colors": [
                "#91CAE8",
                "orange",
                "lightcoral",
                "lightgreen",
                "lightblue",
                "gold",
            ],
            "linestyles": ["--", "-.", ":", "-", "--", "-."],
            "line_width": 2,
            "grid_alpha": 0.3,
            "font_size": 14,
        }

    def _process_dataframe(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Return a copy of `df` enriched with derived indicator columns."""
        df_processed = df.copy()
        for indicator_name, calc_func in self.custom_indicators.items():
            try:
                df_processed[indicator_name] = calc_func(df_processed)
            except Exception as e:
                print(
                    f"Warning: failed to calculate {indicator_name} for {name}: {e}"
                )
        return df_processed

    def _load_csv(self, csv_path: str, name: str) -> Optional[pd.DataFrame]:
        """Load a CSV file and compute derived indicators."""
        try:
            df = pd.read_csv(csv_path)
            df = self._process_dataframe(df, name)
            print(f"Loaded {name}: {len(df)} rows")
            return df
        except FileNotFoundError:
            print(f"File not found: {csv_path}")
            return None
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            return None

    def _create_plot(
        self, title: str, xlabel: str = "Timestep", ylabel: str = "Value"
    ):
        """Create a Matplotlib figure with standard styling."""
        fig, ax = plt.subplots(figsize=self.default_style["figure_size"])
        ax.set_xlabel(xlabel, fontsize=self.default_style["font_size"])
        ax.set_ylabel(ylabel, fontsize=self.default_style["font_size"])
        ax.grid(True, alpha=self.default_style["grid_alpha"])
        ax.set_title(title, fontsize=self.default_style["font_size"])
        return fig, ax

    def _save_or_show(self, fig, output_path: Optional[str] = None):
        """Persist the figure to disk or display it interactively."""
        plt.tight_layout()
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Plot saved: {output_path}")
        else:
            plt.show()

    def _annotate_last_value(
        self, ax: plt.Axes, df: pd.DataFrame, column: str, color: str
    ) -> None:
        """Annotate the last value of `column` for quick visual inspection."""
        if column not in df.columns or "timestep" not in df.columns or df.empty:
            return
        final_value = df[column].iloc[-1]
        annotation = (
            f"{int(final_value)}"
            if isinstance(final_value, (int, np.integer))
            else f"{final_value:.1f}"
        )
        ax.annotate(
            annotation,
            xy=(df["timestep"].iloc[-1], final_value),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=color,
            fontsize=self.default_style["font_size"] - 2,
        )
    
    def plot_fraud_data(
        self,
        data_sources: Dict[str, Union[str, pd.DataFrame]],
        indicators: List[str],
        output_dir: Optional[str] = None,
        mode: str = "compare_sources",
    ) -> None:
        """
        Plot fraud metrics for one or more data sources.

        Args:
            data_sources: Mapping from run label to CSV path or in-memory DataFrame.
            indicators: Metrics to plot.
            output_dir: Directory to save plots. If None, display interactively.
            mode: Either "compare_sources" or "compare_indicators".
        """
        all_data = {}
        for name, source in data_sources.items():
            df = None
            if isinstance(source, str):
                df = self._load_csv(source, name)
            elif isinstance(source, pd.DataFrame):
                df = self._process_dataframe(source, name)
                print(f"Loaded DataFrame '{name}': {len(df)} rows")
            else:
                print(f"Invalid data source type for '{name}': {type(source)}")

            if df is not None:
                all_data[name] = df

        if not all_data:
            print("No valid data found")
            return

        if mode == "compare_sources":
            self._plot_compare_sources(all_data, indicators, output_dir)
        elif mode == "compare_indicators":
            self._plot_compare_indicators(all_data, indicators, output_dir)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'compare_sources' or 'compare_indicators'")

    def _plot_compare_sources(
        self,
        all_data: Dict[str, pd.DataFrame],
        indicators: List[str],
        output_dir: Optional[str],
    ) -> None:
        """Plot each indicator separately while comparing across sources."""
        for indicator in indicators:
            valid_sources = []
            for source_name, df in all_data.items():
                if indicator in df.columns:
                    valid_sources.append(source_name)
                else:
                    print(f"{indicator} not found in {source_name}")

            if not valid_sources:
                print(f"{indicator} not found in any data source")
                continue

            fig, ax = self._create_plot(
                title=f"{indicator.replace('_', ' ').title()} Comparison",
            )

            for idx, source_name in enumerate(valid_sources):
                df = all_data[source_name]
                color = self.default_style["colors"][
                    idx % len(self.default_style["colors"])
                ]
                linestyle = self.default_style["linestyles"][
                    idx % len(self.default_style["linestyles"])
                ]

                ax.plot(
                    df["timestep"],
                    df[indicator],
                    label=source_name,
                    color=color,
                    linestyle=linestyle,
                    linewidth=self.default_style["line_width"],
                )
                self._annotate_last_value(ax, df, indicator, color)

            ax.legend(fontsize=self.default_style["font_size"])
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"{indicator}.png")
            self._save_or_show(fig, output_path)

    def _plot_compare_indicators(
        self,
        all_data: Dict[str, pd.DataFrame],
        indicators: List[str],
        output_dir: Optional[str],
    ) -> None:
        """Plot each source separately while comparing multiple indicators."""
        for source_name, df in all_data.items():
            valid_indicators = [ind for ind in indicators if ind in df.columns]
            missing_indicators = [
                ind for ind in indicators if ind not in df.columns
            ]

            if missing_indicators:
                print(f"{source_name} missing indicators: {missing_indicators}")

            if not valid_indicators:
                print(f"No valid indicators found in {source_name}")
                continue

            fig, ax = self._create_plot(
                title=f"{source_name} - Multiple Indicators",
            )

            for idx, indicator in enumerate(valid_indicators):
                color = self.default_style["colors"][
                    idx % len(self.default_style["colors"])
                ]
                linestyle = self.default_style["linestyles"][
                    idx % len(self.default_style["linestyles"])
                ]

                ax.plot(
                    df["timestep"],
                    df[indicator],
                    label=indicator.replace("_", " ").title(),
                    color=color,
                    linestyle=linestyle,
                    linewidth=self.default_style["line_width"],
                )
                self._annotate_last_value(ax, df, indicator, color)

            ax.legend(fontsize=self.default_style["font_size"])
            output_path = None
            if output_dir:
                output_path = os.path.join(
                    output_dir, f"{source_name}_indicators.png"
                )
            self._save_or_show(fig, output_path)
    
    def quick_fraud_overview(
        self, data_sources: Dict[str, str], output_dir: str
    ) -> None:
        """Generate a ready-made overview plot bundle for common indicators."""
        key_indicators = [
            "private_transfer_money",
            "bad_good_convos",
            "fraud_success_rate",
            "total_likes",
            "total_reposts",
            "total_good_comments",
        ]

        print("Generating fraud data overview...")
        self.plot_fraud_data(
            data_sources=data_sources,
            indicators=key_indicators,
            output_dir=output_dir,
            mode="compare_sources",
        )
        print("Overview generation complete.")

    def calculate_final_stats(self, csv_path: str) -> Dict[str, float]:
        """Return summary statistics using the final timestep of the CSV."""
        df = self._load_csv(csv_path, "stats")
        if df is None:
            return {}

        last_row = df.iloc[-1]
        stats = {
            "private_transfers": last_row.get("private_transfer_money", 0),
            "total_conversations": last_row.get("bad_good_convos", 0),
            "avg_message_depth": last_row.get(
                "average_private_message_depth", 0.0
            ),
            "fraud_success_rate": last_row.get("fraud_success_rate", 0.0),
        }

        print("\nFinal statistics:")
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

        return stats


def create_visualizer() -> FraudDataVisualizer:
    """Instantiate a FraudDataVisualizer with defaults."""
    return FraudDataVisualizer()


def quick_plot(
    data_sources: Dict[str, str],
    indicators: List[str],
    output_dir: Optional[str] = None,
) -> None:
    """Convenience wrapper for one-off plots."""
    viz = create_visualizer()
    viz.plot_fraud_data(data_sources, indicators, output_dir)

if __name__ == "__main__":
    viz = create_visualizer()

    scale_comparison_csv = {
        "1:10": "multiAgent4Fraud/results/different_ratio/1:10.csv",
        "1:20": "multiAgent4Fraud/results/different_ratio/1:20.csv",
        "1:50": "multiAgent4Fraud/results/different_ratio/1:50.csv",
    }

    viz.plot_fraud_data(
        data_sources=scale_comparison_csv,
        indicators=[
            "private_transfer_money",
            "fraud_success_rate",
            "bad_good_convos",
            "total_fraud",
        ],
        output_dir="./outputs/scale_comparison",
        mode="compare_sources",
    )

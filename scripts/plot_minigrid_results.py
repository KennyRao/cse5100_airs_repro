# scripts/plot_minigrid_results.py
import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_run_csv(path: str) -> pd.DataFrame:
    """
    Load one run CSV and return DataFrame with:
    columns: update, global_step, mean_return, policy_loss, value_loss, entropy, arm
    """
    df = pd.read_csv(path)
    # Some runs may have NaNs in mean_return; keep them, we'll handle in plotting.
    return df


def select_run_file(env_tag: str, mode: str, csv_root: str) -> str:
    """
    Pick the correct CSV file for a given env + mode,
    based on the naming convention in train_minigrid.py:

        run_name = f"{env_tag}_{mode}_seed{seed}_{exp_name}"

    where exp_name is usually:
      - 'baseline' for a2c
      - 're3'      for a2c_re3
      - 'airs'     for airs
    """
    # Map each mode to the expected exp_name suffix
    exp_suffix_by_mode = {
        "a2c": "baseline",
        "a2c_re3": "re3",
        "airs": "airs",
    }

    if mode not in exp_suffix_by_mode:
        raise ValueError(f"Unknown mode {mode}")

    exp_suffix = exp_suffix_by_mode[mode]

    # e.g. "Empty-16x16_a2c_seed" and ends with "_baseline.csv"
    prefix = f"{env_tag}_{mode}_seed"
    suffix = f"_{exp_suffix}.csv"

    candidates = [
        f for f in os.listdir(csv_root)
        if f.startswith(prefix) and f.endswith(suffix)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found for env={env_tag}, mode={mode}, "
            f"prefix={prefix}, suffix={suffix}"
        )

    # If later have multiple seeds for the same (env, mode, exp_name),
    # loop over them; for now just take the first in sorted order.
    candidates.sort()
    filename = candidates[0]
    return os.path.join(csv_root, filename)


def smooth_series(series, window: int = 5):
    """
    Simple moving average smoothing. window is in #updates.
    """
    return series.rolling(window=window, min_periods=1).mean()


def plot_env(
    env_tag: str,
    csv_root: str,
    ax: plt.Axes,
    modes: Dict[str, str],
    smooth_window: int = 5,
):
    for mode, label in modes.items():
        try:
            path = select_run_file(env_tag, mode, csv_root)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        df = load_run_csv(path)

        # Use global_step on x-axis, mean_return on y-axis
        x = df["global_step"]
        y = df["mean_return"]

        # Drop NaNs for plotting (episodes might not finish every update)
        mask = y.notna()
        x = x[mask]
        y = y[mask]

        y_smooth = smooth_series(y, window=smooth_window)

        ax.plot(x, y_smooth, label=label)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean Episode Return (moving avg)")
    ax.set_title(env_tag)
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where train_minigrid wrote CSV files.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="Moving average window over updates.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="minigrid_comparison.png",
        help="Path to save the comparison figure.",
    )
    args = parser.parse_args()

    modes = {
        "a2c": "A2C",
        "a2c_re3": "A2C + RE3",
        "airs": "A2C + AIRS",
    }

    # These tags match how we built run_name: "Empty-16x16", "DoorKey-6x6"
    env_tags = ["Empty-16x16", "DoorKey-6x6"]

    fig, axes = plt.subplots(1, len(env_tags), figsize=(12, 4), sharey=True)

    if len(env_tags) == 1:
        axes = [axes]

    for ax, env_tag in zip(axes, env_tags):
        plot_env(
            env_tag=env_tag,
            csv_root=args.results_dir,
            ax=ax,
            modes=modes,
            smooth_window=args.smooth_window,
        )

    plt.tight_layout()
    fig.savefig(args.out_path)
    print(f"Saved figure to {args.out_path}")


if __name__ == "__main__":
    main()

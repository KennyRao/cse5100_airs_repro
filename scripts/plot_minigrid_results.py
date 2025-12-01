# scripts/plot_minigrid_results.py
import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_run_csv(path: str) -> pd.DataFrame:
    """
    Load one run CSV and return DataFrame
    """
    df = pd.read_csv(path)
    return df


def select_run_file(env_tag: str, mode: str, csv_root: str, exp_name: str) -> str:
    """
    Select CSV for given env, mode, and exp_name.
    
    run_name format (from train_minigrid.py):
      f"{env_tag}_{mode}_seed{seed}_{exp_name}.csv"
    """
    prefix = f"{env_tag}_{mode}_seed"
    suffix = f"_{exp_name}.csv"

    candidates = [
        f for f in os.listdir(csv_root)
        if f.startswith(prefix) and f.endswith(suffix)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found for env={env_tag}, mode={mode}, "
            f"exp_name={exp_name}, prefix={prefix}, suffix={suffix}"
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
    a2c_exp: str,
    re3_exp: str,
    rise_exp: str,
    airs_exp_names: List[str],
    smooth_window: int = 5,
):
    # Baselines
    modes_and_exps = [
        ("a2c", a2c_exp, "A2C"),
        ("a2c_re3", re3_exp, "A2C + RE3"),
        ("a2c_rise", rise_exp, "A2C + RISE"),
    ]

    for mode, exp_name, label in modes_and_exps:
        try:
            path = select_run_file(env_tag, mode, csv_root, exp_name)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        df = load_run_csv(path)
        x = df["global_step"]
        y = df["mean_return"]

        mask = y.notna()
        x = x[mask]
        y = y[mask]

        y_smooth = smooth_series(y, window=smooth_window)
        ax.plot(x, y_smooth, label=label)

    # AIRS variants
    for exp_name in airs_exp_names:
        try:
            path = select_run_file(env_tag, "airs", csv_root, exp_name)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        df = load_run_csv(path)
        x = df["global_step"]
        y = df["mean_return"]

        mask = y.notna()
        x = x[mask]
        y = y[mask]

        y_smooth = smooth_series(y, window=smooth_window)

        # Label based on exp_name
        label = f"A2C + AIRS ({exp_name})"

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
    parser.add_argument(
        "--airs_exp_names",
        type=str,
        nargs="+",
        default=["airs_nocost", "airs_cost0.1"],
        help="List of exp_name strings for AIRS variants to plot.",
    )
    parser.add_argument(
        "--a2c_exp",
        type=str,
        default="baseline",
        help="exp_name used for plain A2C runs.",
    )
    parser.add_argument(
        "--re3_exp",
        type=str,
        default="re3",
        help="exp_name used for A2C+RE3 runs.",
    )
    parser.add_argument(
        "--rise_exp",
        type=str,
        default="rise",
        help="exp_name used for A2C+RISE runs.",
    )
    args = parser.parse_args()

    env_tags = ["Empty-16x16", "DoorKey-6x6"]
    fig, axes = plt.subplots(1, len(env_tags), figsize=(12, 4), sharey=True)

    if len(env_tags) == 1:
        axes = [axes]

    for ax, env_tag in zip(axes, env_tags):
        plot_env(
            env_tag=env_tag,
            csv_root=args.results_dir,
            ax=ax,
            a2c_exp=args.a2c_exp,
            re3_exp=args.re3_exp,
            rise_exp=args.rise_exp,
            airs_exp_names=args.airs_exp_names,
            smooth_window=args.smooth_window,
        )

    plt.tight_layout()
    fig.savefig(args.out_path)
    print(f"Saved figure to {args.out_path}")


if __name__ == "__main__":
    main()

# scripts/plot_minigrid_report_figures.py
import argparse
import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def env_id_to_env_tag(env_id: str) -> str:
    """
    Match train_minigrid's env_tag:
      MiniGrid-Empty-16x16-v0 -> Empty-16x16
      MiniGrid-DoorKey-6x6-v0 -> DoorKey-6x6
    """
    return env_id.replace("MiniGrid-", "").replace("-v0", "")


def format_airs_exp_name(cost_penalty: float) -> str:
    """
    exp_name like: airs_cost0, airs_cost0.05, airs_cost0.1
    using :g formatting.
    """
    return f"airs_cost{cost_penalty:g}"


def load_run_csv(
    results_dir: str,
    env_tag: str,
    mode: str,
    seed: int,
    exp_name: str,
) -> Optional[pd.DataFrame]:
    """
    Load a single run CSV given env_tag, mode, seed, exp_name.
    File name format from train_minigrid:
      f"{env_tag}_{mode}_seed{seed}_{exp_name}.csv"
    """
    filename = f"{env_tag}_{mode}_seed{seed}_{exp_name}.csv"
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        print(f"[WARN] Missing CSV: {path}")
        return None
    df = pd.read_csv(path)
    return df


def aggregate_mean_return(
    results_dir: str,
    env_tag: str,
    mode: str,
    exp_name: str,
    seeds: List[int],
    smooth_window: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Aggregate mean_return across multiple seeds for one (env_tag, mode, exp_name).
    Returns (x, mean_y, std_y) or None if no runs found.
    """
    xs = []
    ys = []

    for seed in seeds:
        df = load_run_csv(results_dir, env_tag, mode, seed, exp_name)
        if df is None:
            continue

        # Drop NaNs in mean_return
        y = df["mean_return"]
        mask = y.notna()
        x = df["global_step"][mask].to_numpy()
        y = y[mask]

        # Simple moving average smoothing in "update space"
        y_smooth = y.rolling(window=smooth_window, min_periods=1).mean().to_numpy()

        xs.append(x)
        ys.append(y_smooth)

    if not xs:
        print(
            f"[WARN] No runs found for env={env_tag}, mode={mode}, exp={exp_name}."
        )
        return None

    # Align lengths by truncating to the minimum length across seeds
    min_len = min(len(x) for x in xs)
    x0 = xs[0][:min_len]
    y_stack = np.stack([y[:min_len] for y in ys], axis=0)

    mean_y = y_stack.mean(axis=0)
    std_y = y_stack.std(axis=0)

    return x0, mean_y, std_y


def aggregate_steps_per_sec(
    results_dir: str,
    env_tag: str,
    exp_name: str,
    seeds: List[int],
    mode: str = "airs",
) -> Optional[Tuple[float, float]]:
    """
    Aggregate steps_per_sec across seeds for a given AIRS exp_name.
    Returns (mean_steps_per_sec, std_steps_per_sec) or None.
    """
    values = []

    for seed in seeds:
        df = load_run_csv(results_dir, env_tag, mode, seed, exp_name)
        if df is None:
            continue

        if "steps_per_sec" not in df.columns:
            print(
                f"[WARN] steps_per_sec not found in CSV for {env_tag}, {mode}, seed={seed}, exp={exp_name}."
            )
            continue

        # Average over updates for this seed
        v = df["steps_per_sec"].mean()
        values.append(v)

    if not values:
        return None

    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_main_learning_curves(
    results_dir: str,
    plots_dir: str,
    env_ids: List[str],
    seeds: List[int],
    airs_cost_penalties: List[float],
    smooth_window: int,
):
    """
    Create a single figure with 1 subplot per env:
      A2C, A2C+RE3, A2C+RISE, AIRS(λ=...) curves (mean ± std across seeds).
    """
    ensure_dir(plots_dir)
    env_tags = [env_id_to_env_tag(e) for e in env_ids]

    fig, axes = plt.subplots(
        1, len(env_tags), figsize=(6 * len(env_tags), 4), sharey=True
    )
    if len(env_tags) == 1:
        axes = [axes]

    baseline_configs = [
        ("a2c", "baseline", "A2C"),
        ("a2c_re3", "re3", "A2C + RE3"),
        ("a2c_rise", "rise", "A2C + RISE"),
    ]

    for ax, env_tag, env_id in zip(axes, env_tags, env_ids):
        # Baselines
        for mode, exp_name, label in baseline_configs:
            agg = aggregate_mean_return(
                results_dir=results_dir,
                env_tag=env_tag,
                mode=mode,
                exp_name=exp_name,
                seeds=seeds,
                smooth_window=smooth_window,
            )
            if agg is None:
                continue
            x, mean_y, std_y = agg
            ax.plot(x, mean_y, label=label)
            ax.fill_between(
                x, mean_y - std_y, mean_y + std_y, alpha=0.2, linewidth=0
            )

        # AIRS variants for each λ
        for lam in airs_cost_penalties:
            exp_name = format_airs_exp_name(lam)
            label = f"AIRS (λ={lam:g})"
            agg = aggregate_mean_return(
                results_dir=results_dir,
                env_tag=env_tag,
                mode="airs",
                exp_name=exp_name,
                seeds=seeds,
                smooth_window=smooth_window,
            )
            if agg is None:
                continue
            x, mean_y, std_y = agg
            ax.plot(x, mean_y, label=label)
            ax.fill_between(
                x, mean_y - std_y, mean_y + std_y, alpha=0.2, linewidth=0
            )

        ax.set_title(env_tag)
        ax.set_xlabel("Environment Steps")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean Episode Return (mean ± std)")
    axes[0].legend(loc="best")

    fig.tight_layout()
    out_path = os.path.join(plots_dir, "minigrid_main_learning_curves.png")
    fig.savefig(out_path, dpi=200)
    print(f"[SAVE] {out_path}")


def plot_steps_per_sec_vs_lambda(
    results_dir: str,
    plots_dir: str,
    env_ids: List[str],
    seeds: List[int],
    airs_cost_penalties: List[float],
):
    """
    For each env, create a bar chart:
      x-axis: λ values
      y-axis: mean steps_per_sec across seeds
      error bars: std across seeds.
    """
    ensure_dir(plots_dir)
    env_tags = [env_id_to_env_tag(e) for e in env_ids]

    for env_tag, env_id in zip(env_tags, env_ids):
        means = []
        stds = []
        labels = []

        for lam in airs_cost_penalties:
            exp_name = format_airs_exp_name(lam)
            agg = aggregate_steps_per_sec(
                results_dir=results_dir,
                env_tag=env_tag,
                exp_name=exp_name,
                seeds=seeds,
                mode="airs",
            )
            if agg is None:
                print(
                    f"[WARN] No steps_per_sec data for env={env_tag}, λ={lam:g}"
                )
                continue
            m, s = agg
            means.append(m)
            stds.append(s)
            labels.append(f"{lam:g}")

        if not means:
            continue

        x = np.arange(len(means))

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(x, means, yerr=stds, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("λ (cost penalty)")
        ax.set_ylabel("Mean steps per second")
        ax.set_title(f"{env_tag} – AIRS speed vs λ")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        out_path = os.path.join(
            plots_dir, f"{env_tag}_airs_steps_per_sec_vs_lambda.png"
        )
        fig.savefig(out_path, dpi=200)
        print(f"[SAVE] {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate report-ready plots for MiniGrid AIRS experiments "
            "(multi-seed curves and cost-awareness ablations)."
        )
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory with CSV result files from train_minigrid.",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="plots/minigrid",
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--env_ids",
        type=str,
        nargs="*",
        default=[
            "MiniGrid-Empty-16x16-v0",
            "MiniGrid-DoorKey-6x6-v0",
        ],
        help="MiniGrid env IDs to include in plots.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="Seeds to aggregate over (must match the runs you trained).",
    )
    parser.add_argument(
        "--airs_cost_penalties",
        type=float,
        nargs="*",
        default=[0.0, 0.05, 0.1, 0.2],
        help="λ values used for AIRS cost-penalty experiments.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="Moving-average window (in updates) for smoothing learning curves.",
    )

    args = parser.parse_args()

    print(f"Using results_dir={args.results_dir}")
    print(f"Saving plots under plots_dir={args.plots_dir}")

    plot_main_learning_curves(
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        env_ids=args.env_ids,
        seeds=args.seeds,
        airs_cost_penalties=args.airs_cost_penalties,
        smooth_window=args.smooth_window,
    )

    plot_steps_per_sec_vs_lambda(
        results_dir=args.results_dir,
        plots_dir=args.plots_dir,
        env_ids=args.env_ids,
        seeds=args.seeds,
        airs_cost_penalties=args.airs_cost_penalties,
    )


if __name__ == "__main__":
    main()

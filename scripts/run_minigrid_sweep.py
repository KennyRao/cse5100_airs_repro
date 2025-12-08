# scripts/run_minigrid_sweep.py
import argparse
import csv
import os
import shutil
from typing import List, Tuple, Optional
import torch
from scripts.train_minigrid import TrainConfig, train


def env_id_to_env_tag(env_id: str) -> str:
    """
    Match the env_tag logic in train_minigrid.train():
    MiniGrid-Empty-16x16-v0 -> Empty-16x16
    MiniGrid-DoorKey-6x6-v0 -> DoorKey-6x6
    """
    return env_id.replace("MiniGrid-", "").replace("-v0", "")


def format_airs_exp_name(cost_penalty: float) -> str:
    """
    exp_name like: airs_cost0, airs_cost0.05, airs_cost0.1
    using :g formatting for nicer strings.
    """
    return f"airs_cost{cost_penalty:g}"


def compute_expected_steps(
    total_timesteps: int, num_envs: int, num_steps: int
) -> Tuple[int, int]:
    """
    train_minigrid computes:
        num_updates = total_timesteps // (num_envs * num_steps)
        global_step increases by num_envs * num_steps each update.
    So final global_step = num_updates * num_envs * num_steps.
    """
    updates = total_timesteps // (num_envs * num_steps)
    expected_steps = updates * num_envs * num_steps
    return expected_steps, updates


def read_last_global_step(csv_path: str) -> Optional[int]:
    """
    Read the last global_step from a CSV file.
    CSV header is:
        update,global_step,mean_return,...
    """
    if not os.path.exists(csv_path):
        return None

    last_line = None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            last_line = row

    if last_line is None:
        return None

    try:
        # index 1 = global_step
        return int(float(last_line[1]))
    except (IndexError, ValueError):
        return None


def is_run_complete(
    results_dir: str,
    run_name: str,
    expected_global_step: int,
    done_on_disk: bool = True,
) -> bool:
    """
    A run is considered COMPLETE if either:
      - a .done file exists (if done_on_disk=True), OR
      - the CSV exists AND its last global_step >= expected_global_step.
    """
    csv_path = os.path.join(results_dir, f"{run_name}.csv")
    done_path = os.path.join(results_dir, f"{run_name}.done")

    # 1) .done flag
    if done_on_disk and os.path.exists(done_path):
        return True

    # 2) CSV + global_step check
    last_global_step = read_last_global_step(csv_path)
    if last_global_step is None:
        return False

    return last_global_step >= expected_global_step


def clean_incomplete_run(
    results_dir: str,
    log_dir: str,
    run_name: str,
):
    """
    Delete partial artifacts for an incomplete run so it can be restarted cleanly.
    """
    csv_path = os.path.join(results_dir, f"{run_name}.csv")
    done_path = os.path.join(results_dir, f"{run_name}.done")
    tb_dir = os.path.join(log_dir, run_name)

    if os.path.exists(csv_path):
        print(f"  - Deleting partial CSV: {csv_path}")
        os.remove(csv_path)

    if os.path.exists(done_path):
        print(f"  - Deleting stale .done file: {done_path}")
        os.remove(done_path)

    if os.path.exists(tb_dir):
        print(f"  - Deleting partial TensorBoard directory: {tb_dir}")
        shutil.rmtree(tb_dir, ignore_errors=True)


def mark_run_done(results_dir: str, run_name: str):
    """
    Create a small .done marker file once training finishes successfully.
    """
    done_path = os.path.join(results_dir, f"{run_name}.done")
    with open(done_path, "w", encoding="utf-8") as f:
        f.write("done\n")


def run_single_experiment(
    env_id: str,
    mode: str,
    seed: int,
    exp_name: str,
    airs_cost_penalty: float,
    args,
    expected_global_step: int,
    expected_updates: int,
):
    """
    Check status and run one (env, mode, seed, exp_name, lambda) combo.
    """
    env_tag = env_id_to_env_tag(env_id)
    run_name = f"{env_tag}_{mode}_seed{seed}_{exp_name}"

    print("\n========================")
    print(f"Experiment: {run_name}")
    print("========================")

    # Check completion
    if is_run_complete(args.results_dir, run_name, expected_global_step):
        last_step = read_last_global_step(
            os.path.join(args.results_dir, f"{run_name}.csv")
        )
        print(f"[SKIP] Run already complete (last global_step={last_step}).")
        return

    # If CSV exists but not complete, clean it up
    csv_path = os.path.join(args.results_dir, f"{run_name}.csv")
    if os.path.exists(csv_path):
        print("[INFO] Found partial CSV; treating as incomplete and resetting.")
        clean_incomplete_run(args.results_dir, args.log_dir, run_name)

    # Build TrainConfig for this experiment
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = TrainConfig(
        env_id=env_id,
        mode=mode,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        re3_k=args.re3_k,
        re3_beta0_empty=args.re3_beta0_empty,
        re3_beta0_door=args.re3_beta0_door,
        re3_kappa=args.re3_kappa,
        rise_alpha=args.rise_alpha,
        airs_c=args.airs_c,
        airs_window=args.airs_window,
        airs_cost_penalty=airs_cost_penalty,
        seed=seed,
        device=device,
        log_dir=args.log_dir,
        results_dir=args.results_dir,
        exp_name=exp_name,
        log_interval_updates=args.log_interval_updates,
    )

    # Run training
    print(
        f"[RUN ] Starting training: env={env_id}, mode={mode}, seed={seed}, "
        f"exp_name={exp_name}, lambda={airs_cost_penalty}"
    )
    print(
        f"       Expected updates={expected_updates}, "
        f"expected final global_step={expected_global_step}"
    )
    train(cfg)

    # Mark complete
    mark_run_done(args.results_dir, run_name)

    # Sanity check
    last_step = read_last_global_step(csv_path)
    if last_step is None or last_step < expected_global_step:
        print(
            f"[WARN] After training, last global_step={last_step} < expected={expected_global_step}."
        )
    else:
        print(f"[OK  ] Finished {run_name} (last global_step={last_step}).")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a full MiniGrid experiment sweep (envs x modes x seeds x "
            "AIRS cost penalties) with crash-safe skipping of completed runs."
        )
    )

    # Which environments / seeds / cost penalties to run
    parser.add_argument(
        "--env_ids",
        type=str,
        nargs="*",
        default=[
            "MiniGrid-Empty-16x16-v0",
            "MiniGrid-DoorKey-6x6-v0",
        ],
        help=(
            "MiniGrid environment IDs to run. "
            "Default: Empty-16x16 and DoorKey-6x6."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="List of random seeds to run for each configuration.",
    )
    parser.add_argument(
        "--airs_cost_penalties",
        type=float,
        nargs="*",
        default=[0.0, 0.05, 0.1, 0.2],
        help=(
            "List of λ values for AIRS cost penalty. "
            "λ=0.0 reproduces original AIRS; >0.0 is cost-aware."
        ),
    )

    # Core training hyperparameters (shared across all experiments)
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1_000_000,
        help="Total env steps across all envs (per run).",
    )
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    # Intrinsic reward hyperparams
    parser.add_argument("--re3_k", type=int, default=3)
    parser.add_argument("--re3_beta0_empty", type=float, default=0.1)
    parser.add_argument("--re3_beta0_door", type=float, default=0.005)
    parser.add_argument("--re3_kappa", type=float, default=0.0)
    parser.add_argument("--rise_alpha", type=float, default=0.5)

    # AIRS bandit hyperparams
    parser.add_argument("--airs_c", type=float, default=1.0)
    parser.add_argument("--airs_window", type=int, default=10)

    # Logging / paths
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs",
        help="TensorBoard log directory root.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to store CSV result files and .done flags.",
    )
    parser.add_argument(
        "--log_interval_updates",
        type=int,
        default=10,
        help="How often (in updates) to print progress from train_minigrid.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. 'cpu' or 'cuda'. If None, auto-detect.",
    )

    args = parser.parse_args()

    # Compute expected final global_step and number of updates
    expected_global_step, expected_updates = compute_expected_steps(
        args.total_timesteps, args.num_envs, args.num_steps
    )
    print(
        f"Global settings: total_timesteps={args.total_timesteps}, "
        f"num_envs={args.num_envs}, num_steps={args.num_steps}"
    )
    print(
        f"Derived: num_updates={expected_updates}, "
        f"expected final global_step={expected_global_step}"
    )

    # Sweep over envs, seeds, and modes
    for env_id in args.env_ids:
        print("\n========================================")
        print(f"Environment: {env_id}")
        print("========================================")
        for seed in args.seeds:
            # Baselines: A2C, A2C+RE3, A2C+RISE
            run_single_experiment(
                env_id=env_id,
                mode="a2c",
                seed=seed,
                exp_name="baseline",
                airs_cost_penalty=0.0,
                args=args,
                expected_global_step=expected_global_step,
                expected_updates=expected_updates,
            )
            run_single_experiment(
                env_id=env_id,
                mode="a2c_re3",
                seed=seed,
                exp_name="re3",
                airs_cost_penalty=0.0,
                args=args,
                expected_global_step=expected_global_step,
                expected_updates=expected_updates,
            )
            run_single_experiment(
                env_id=env_id,
                mode="a2c_rise",
                seed=seed,
                exp_name="rise",
                airs_cost_penalty=0.0,
                args=args,
                expected_global_step=expected_global_step,
                expected_updates=expected_updates,
            )

            # AIRS + cost sweep
            for lam in args.airs_cost_penalties:
                exp_name = format_airs_exp_name(lam)
                run_single_experiment(
                    env_id=env_id,
                    mode="airs",
                    seed=seed,
                    exp_name=exp_name,
                    airs_cost_penalty=lam,
                    args=args,
                    expected_global_step=expected_global_step,
                    expected_updates=expected_updates,
                )


if __name__ == "__main__":
    main()

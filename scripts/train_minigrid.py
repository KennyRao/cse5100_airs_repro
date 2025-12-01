# scripts/train_minigrid.py
import argparse
import os
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import gymnasium as gym
import minigrid

from airs.networks import ActorCriticNet, RandomEncoder
from airs.intrinsic import IdentityIntrinsic, RE3Intrinsic, RISEIntrinsic, beta_schedule
from airs.bandit import UCBIntrinsicBandit

import time


@dataclass
class TrainConfig:
    env_id: str
    mode: str  # "a2c", "a2c_re3", "a2c_rise", "airs"

    total_timesteps: int = 200_000
    num_envs: int = 16  # number of parallel envs
    num_steps: int = 128  # steps per rollout
    gamma: float = 0.99  # discount factor
    gae_lambda: float = 0.95

    learning_rate: float = 2.5e-4
    ent_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Intrinsic reward hyperparams (MiniGrid from AIRS paper)
    re3_k: int = 3
    re3_beta0_empty: float = 0.1
    re3_beta0_door: float = 0.005
    re3_kappa: float = 0.0

    # RISE hyperparam
    rise_alpha: float = 0.5

    # AIRS bandit hyperparams
    airs_c: float = 1.0
    airs_window: int = 10
    airs_cost_penalty: float = 0.0

    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging / saving
    log_dir: str = "runs"
    results_dir: str = "results"
    exp_name: str = "default"
    log_interval_updates: int = 10


def make_env(env_id: str, seed: int):
    """
    Returns a thunk to create a MiniGrid environment.
    We use fully observed image observations.
    """
    from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

    def thunk():
        env = gym.make(env_id)
        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        env.reset(seed=seed)
        return env

    return thunk


def preprocess_obs(obs: np.ndarray) -> torch.Tensor:
    """
    MiniGrid ImgObsWrapper returns [H, W, C] uint8 images.
    We want [B, C, H, W] float32 in [0,1].
    """
    # obs: [B, H, W, C]
    obs_t = torch.from_numpy(obs).float() / 255.0
    obs_t = obs_t.permute(0, 3, 1, 2)  # to [B, C, H, W]
    return obs_t


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    rewards: [T, N]
    values: [T, N]
    dones: [T, N] float {0,1}
    last_value: [N]
    Returns (returns, advantages) both [T, N]
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        if t == T - 1:
            next_values = last_value
        else:
            next_values = values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return returns, advantages


def train(config: TrainConfig):
    device = torch.device(config.device)

    # Pretty run name safe for paths
    env_tag = config.env_id.replace("MiniGrid-", "").replace("-v0", "")
    run_name = f"{env_tag}_{config.mode}_seed{config.seed}_{config.exp_name}"

    # Make dirs
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)

    # TensorBoard writer
    tb_path = os.path.join(config.log_dir, run_name)
    writer = SummaryWriter(log_dir=tb_path)

    # CSV results file (for plotting later)
    csv_path = os.path.join(config.results_dir, f"{run_name}.csv")
    csv_fp = open(csv_path, "w", encoding="utf-8")
    csv_fp.write(
        "update,global_step,mean_return,policy_loss,value_loss,entropy,arm,"
        "task_return_est,mean_ext_reward,mean_int_reward,steps_per_sec,"
        "cost_id,cost_re3,cost_rise\n"
    )

    # Vectorized env
    envs = gym.vector.SyncVectorEnv(
        [make_env(config.env_id, config.seed + i) for i in range(config.num_envs)]
    )
    obs, _ = envs.reset(seed=config.seed)
    obs_t = preprocess_obs(obs)
    obs_shape = obs_t.shape[1:]  # [C,H,W]
    num_actions = envs.single_action_space.n

    # Networks
    ac_net = ActorCriticNet(obs_shape, num_actions).to(device)
    rnd_encoder = RandomEncoder(obs_shape).to(device)

    optimizer = torch.optim.Adam(ac_net.parameters(), lr=config.learning_rate)

    # Intrinsic modules
    id_intrinsic = IdentityIntrinsic(device)
    re3_intrinsic = RE3Intrinsic(
        encoder=rnd_encoder,
        device=device,
        k=config.re3_k,
        max_buffer_size=10000,
    )
    rise_intrinsic = RISEIntrinsic(
        encoder=rnd_encoder,
        device=device,
        alpha=config.rise_alpha,
        k=config.re3_k,
        max_buffer_size=10000,
    )

    # Bandit for AIRS
    if config.mode == "airs":
        bandit = UCBIntrinsicBandit(
            arms=["id", "re3", "rise"],
            c=config.airs_c,
            window=config.airs_window,
            cost_penalty=config.airs_cost_penalty,
            arm_costs=None,  # will be learned from runtime
        )
    else:
        bandit = None

    global_step = 0
    start_update = 0

    total_steps = config.total_timesteps
    num_updates = total_steps // (config.num_envs * config.num_steps)

    # Choose beta0 depending on env
    if "Empty-16x16" in config.env_id:
        beta0 = config.re3_beta0_empty
    elif "DoorKey-6x6" in config.env_id:
        beta0 = config.re3_beta0_door
    else:
        beta0 = config.re3_beta0_empty  # fallback

    print(f"Training {config.mode} on {config.env_id} for {total_steps} steps")
    print(f"Run name: {run_name}")
    print(f"TensorBoard logdir: {tb_path}")
    print(f"CSV path: {csv_path}")

    # Stats
    episode_returns: List[float] = []
    # For AIRS: count how often each arm is selected
    arm_counts = {"id": 0, "re3": 0, "rise": 0}

    for update in tqdm(range(start_update, num_updates)):
        # Measure wall-clock performance per update
        update_start_time = time.perf_counter()

        # For plotting reward composition
        sum_ext_reward = 0.0
        sum_int_reward = 0.0

        # For AIRS: this will be filled when we update the bandit
        task_return_est = float("nan")

        # Decide which intrinsic mode to use for this update
        if config.mode == "a2c":
            intrinsic_arm = "id"
        elif config.mode == "a2c_re3":
            intrinsic_arm = "re3"
        elif config.mode == "a2c_rise":
            intrinsic_arm = "rise"
        elif config.mode == "airs":
            intrinsic_arm = bandit.select_arm()
            arm_counts[intrinsic_arm] += 1
        else:
            raise ValueError(f"Unknown mode {config.mode}")

        # Buffers
        obs_buf = torch.zeros(
            config.num_steps, config.num_envs, *obs_shape, device=device
        )
        actions_buf = torch.zeros(
            config.num_steps, config.num_envs, dtype=torch.long, device=device
        )
        logprobs_buf = torch.zeros(
            config.num_steps, config.num_envs, device=device
        )
        rewards_mix_buf = torch.zeros(
            config.num_steps, config.num_envs, device=device
        )
        rewards_ext_buf = torch.zeros(   # extrinsic (E) only
            config.num_steps, config.num_envs, device=device
        )
        dones_buf = torch.zeros(
            config.num_steps, config.num_envs, device=device
        )
        values_mix_buf = torch.zeros(    # value for E+I
            config.num_steps, config.num_envs, device=device
        )
        values_task_buf = torch.zeros(   # value for E
            config.num_steps, config.num_envs, device=device
        )

        ep_returns_this_update: List[float] = []

        for t in range(config.num_steps):
            global_step += config.num_envs

            # Prepare obs tensor
            obs_torch = preprocess_obs(obs).to(device)

            with torch.no_grad():
                logits, v_total, v_task = ac_net(obs_torch)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logprobs = dist.log_prob(actions)

            next_obs, extrinsic_reward, terminated, truncated, infos = envs.step(
                actions.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            
            # Track episode returns from extrinsic rewards only
            for i, d in enumerate(done):
                if d:
                    final_info = infos.get("final_info")
                    if final_info is not None and final_info[i] is not None:
                        ep_returns_this_update.append(
                            final_info[i]["episode"]["r"]
                        )
                    else:
                        ep_returns_this_update.append(float(extrinsic_reward[i]))

            # Intrinsic reward
            with torch.no_grad():
                # Timing for cost-aware bandit
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                
                if intrinsic_arm == "id":
                    intrinsic = id_intrinsic.compute(obs_torch)
                elif intrinsic_arm == "re3":
                    intrinsic = re3_intrinsic.compute(obs_torch)
                elif intrinsic_arm == "rise":
                    intrinsic = rise_intrinsic.compute(obs_torch)
                else:
                    intrinsic = torch.zeros(config.num_envs, device=device)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                elapsed = t1 - t0
            
            # Only AIRS uses the bandit & runtime measurements
            if config.mode == "airs" and bandit is not None and intrinsic_arm in ["id", "re3", "rise"]:
                bandit.record_cost(intrinsic_arm, elapsed)
            
            extrinsic_t = torch.from_numpy(extrinsic_reward).to(device)
            beta_t = beta_schedule(global_step, beta0, config.re3_kappa)
            total_reward = extrinsic_t + beta_t * intrinsic
            
            # Accumulate per-step reward stats for logging
            sum_ext_reward += extrinsic_t.mean().item()
            sum_int_reward += (beta_t * intrinsic).mean().item()
            
            # Save to buffers
            obs_buf[t] = obs_torch
            actions_buf[t] = actions
            logprobs_buf[t] = logprobs
            rewards_mix_buf[t] = total_reward
            rewards_ext_buf[t] = extrinsic_t
            dones_buf[t] = torch.from_numpy(done.astype(np.float32)).to(device)
            values_mix_buf[t] = v_total  # value for E+I
            values_task_buf[t] = v_task  # value for E
            
            obs = next_obs

        # Compute last value for GAE
        with torch.no_grad():
            last_obs_torch = preprocess_obs(obs).to(device)
            _, last_v_total, last_v_task = ac_net(last_obs_torch)

        # GAE for mixed reward
        returns_mix, advantages = compute_gae(
            rewards_mix_buf,
            values_mix_buf,
            dones_buf,
            last_v_total,
            gamma=config.gamma,
            lam=config.gae_lambda,
        )
        
        # GAE for extrinsic-only reward
        returns_ext, _ = compute_gae(
            rewards_ext_buf,
            values_task_buf,
            dones_buf,
            last_v_task,
            gamma=config.gamma,
            lam=config.gae_lambda,
        )

        # Flatten
        b_obs = obs_buf.reshape(-1, *obs_shape)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        
        b_returns_mix = returns_mix.reshape(-1)
        b_returns_ext = returns_ext.reshape(-1)
        b_advantages = advantages.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

        # A2C update with two value heads
        logits, v_total_pred, v_task_pred = ac_net(b_obs)
        dist = Categorical(logits=logits)
        new_logprobs = dist.log_prob(b_actions)
        entropy = dist.entropy().mean()

        policy_loss = -(b_advantages * new_logprobs).mean()

        value_loss_mix = 0.5 * (b_returns_mix - v_total_pred).pow(2).mean()
        value_loss_task = 0.5 * (b_returns_ext - v_task_pred).pow(2).mean()
        value_loss = value_loss_mix + value_loss_task

        loss = policy_loss + config.value_coef * value_loss - config.ent_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ac_net.parameters(), config.max_grad_norm)
        optimizer.step()

        # Bandit update (AIRS only) using task-return estimator
        if config.mode == "airs" and bandit is not None:
            # One scalar per rollout: average estimated extrinsic return
            task_return_est = float(returns_ext.mean().item())
            bandit.update(intrinsic_arm, task_return_est)
        
        # After this update, refresh the cost estimates based on runtime
        if config.mode == "airs" and bandit is not None:
            bandit.recompute_arm_costs(base_arm="id")
        
        # Compute per-update averages for logging/plotting
        mean_ext_reward = sum_ext_reward / config.num_steps
        mean_int_reward = sum_int_reward / config.num_steps

        # Wall-clock efficiency: steps per second this update
        update_duration = time.perf_counter() - update_start_time
        steps_this_update = config.num_envs * config.num_steps
        steps_per_sec = steps_this_update / max(update_duration, 1e-8)

        # Current cost estimates for each arm (NaN for non-AIRS)
        if config.mode == "airs" and bandit is not None:
            cost_id  = float(bandit.arm_costs.get("id",   float("nan")))
            cost_re3 = float(bandit.arm_costs.get("re3",  float("nan")))
            cost_rise = float(bandit.arm_costs.get("rise", float("nan")))
        else:
            cost_id = cost_re3 = cost_rise = float("nan")

        # Aggregate stats
        if len(ep_returns_this_update) > 0:
            episode_returns.extend(ep_returns_this_update)
            mean_return = float(np.mean(episode_returns[-50:]))
        else:
            mean_return = float("nan")

        # TensorBoard logging
        if not np.isnan(mean_return):
            writer.add_scalar(
                "train/episode_return_mean50", mean_return, global_step
            )
        writer.add_scalar("loss/policy", policy_loss.item(), global_step)
        writer.add_scalar("loss/value", value_loss.item(), global_step)
        writer.add_scalar("loss/entropy", entropy.item(), global_step)
        
        # Reward composition and speed
        writer.add_scalar("reward/mean_ext_per_step", mean_ext_reward, global_step)
        writer.add_scalar("reward/mean_int_per_step", mean_int_reward, global_step)
        writer.add_scalar("speed/steps_per_sec", steps_per_sec, global_step)

        if config.mode == "airs":
            total_arm_picks = sum(arm_counts.values())
            if total_arm_picks > 0:
                for arm in ["id", "re3", "rise"]:
                    frac = arm_counts[arm] / total_arm_picks
                    writer.add_scalar(f"airs/fraction_{arm}", frac, global_step)
            # Log task return estimate
            writer.add_scalar("airs/task_return_est", task_return_est, global_step)
            
            # Log cost estimates
            for arm in ["id", "re3", "rise"]:
                if arm in bandit.arm_costs:
                    writer.add_scalar(f"airs/cost_{arm}", bandit.arm_costs[arm], global_step)

        # CSV logging
        csv_fp.write(
            f"{update},{global_step},{mean_return},"
            f"{policy_loss.item()},{value_loss.item()},{entropy.item()},"
            f"{intrinsic_arm},"
            f"{task_return_est},"
            f"{mean_ext_reward},{mean_int_reward},"
            f"{steps_per_sec},"
            f"{cost_id},{cost_re3},{cost_rise}\n"
        )
        csv_fp.flush()

        # Console logging
        if (update + 1) % config.log_interval_updates == 0:
            print(
                f"Update {update+1}/{num_updates} | "
                f"Mode {config.mode} | "
                f"Mean episode return (last 50) = {mean_return:.3f}"
            )

    if config.mode == "airs":
        print("AIRS arm selection counts:", arm_counts)

    csv_fp.close()
    writer.close()
    envs.close()


def main():
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument(
        "--env_id",
        type=str,
        default="MiniGrid-Empty-16x16-v0",
        help="MiniGrid env id, e.g. MiniGrid-Empty-16x16-v0 or MiniGrid-DoorKey-6x6-v0",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["a2c", "a2c_re3", "a2c_rise", "airs"],
        default="a2c",
        help="Which method to run",
    )

    # randomness
    parser.add_argument(
        "--total_timesteps", type=int, default=1_000_000,
        help="Total env steps across all envs"
    )
    parser.add_argument("--seed", type=int, default=1)

    # Core training hyperparameters
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
    parser.add_argument(
        "--airs_cost_penalty",
        type=float,
        default=0.0,
        help="λ: penalty per unit arm_cost in the bandit UCB score",
    )

    # Logging / paths
    parser.add_argument(
        "--exp_name",
        type=str,
        default="default",
        help="Extra tag to distinguish runs (e.g. date, comment).",
    )
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
        help="Directory to store CSV result files.",
    )
    parser.add_argument(
        "--log_interval_updates",
        type=int,
        default=10,
        help="How often (in updates) to print to console.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. 'cpu' or 'cuda'. "
             "If None, uses CUDA if available.",
    )

    args = parser.parse_args()

    # Determine device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = TrainConfig(
        env_id=args.env_id,
        mode=args.mode,
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
        airs_cost_penalty=args.airs_cost_penalty,
        seed=args.seed,
        device=device,
        log_dir=args.log_dir,
        results_dir=args.results_dir,
        exp_name=args.exp_name,
        log_interval_updates=args.log_interval_updates,
    )
    train(cfg)


if __name__ == "__main__":
    main()

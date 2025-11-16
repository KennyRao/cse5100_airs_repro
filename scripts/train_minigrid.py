# scripts/train_minigrid.py
import argparse
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

import gymnasium as gym
import minigrid

from airs.networks import ActorCriticNet, RandomEncoder
from airs.intrinsic import IdentityIntrinsic, RE3Intrinsic, beta_schedule
from airs.bandit import UCBIntrinsicBandit


@dataclass
class TrainConfig:
    env_id: str
    mode: str  # "a2c", "a2c_re3", "airs"

    total_timesteps: int = 200_000
    num_envs: int = 16
    num_steps: int = 128  # steps per rollout
    gamma: float = 0.99
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

    # AIRS bandit hyperparams
    airs_c: float = 1.0
    airs_window: int = 10

    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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

    # Bandit for AIRS
    if config.mode == "airs":
        bandit = UCBIntrinsicBandit(arms=["id", "re3"], c=config.airs_c,
                                    window=config.airs_window)
    else:
        bandit = None

    global_step = 0
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

    # Stats
    episode_returns: List[float] = []
    # For AIRS: count how often each arm is selected
    arm_counts = {"id": 0, "re3": 0}

    for update in tqdm(range(num_updates)):
        # Decide which intrinsic mode to use for this update
        if config.mode == "a2c":
            intrinsic_arm = "id"
        elif config.mode == "a2c_re3":
            intrinsic_arm = "re3"
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
        dones_buf = torch.zeros(
            config.num_steps, config.num_envs, device=device
        )
        values_buf = torch.zeros(
            config.num_steps, config.num_envs, device=device
        )

        ep_returns_this_update: List[float] = []

        for t in range(config.num_steps):
            global_step += config.num_envs

            # Prepare obs tensor
            obs_torch = preprocess_obs(obs).to(device)

            with torch.no_grad():
                logits, values = ac_net(obs_torch)
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
                    # infos["final_info"] is a list; handle both classic gym vs gymnasium
                    final_info = infos.get("final_info")
                    if final_info is not None and final_info[i] is not None:
                        ep_returns_this_update.append(final_info[i]["episode"]["r"])
                    else:
                        # Fallback: sum extrinsic rewards (MiniGrid reward at end)
                        ep_returns_this_update.append(float(extrinsic_reward[i]))

            # Intrinsic reward
            with torch.no_grad():
                if intrinsic_arm == "id":
                    intrinsic = id_intrinsic.compute(obs_torch)
                elif intrinsic_arm == "re3":
                    intrinsic = re3_intrinsic.compute(obs_torch)
                else:
                    intrinsic = torch.zeros(
                        config.num_envs, device=device
                    )

            beta_t = beta_schedule(global_step, beta0, config.re3_kappa)
            total_reward = torch.from_numpy(extrinsic_reward).to(device) + \
                           beta_t * intrinsic

            # Save to buffers
            obs_buf[t] = obs_torch
            actions_buf[t] = actions
            logprobs_buf[t] = logprobs
            rewards_mix_buf[t] = total_reward
            dones_buf[t] = torch.from_numpy(done.astype(np.float32)).to(device)
            values_buf[t] = values

            obs = next_obs

        # Compute last value for GAE
        with torch.no_grad():
            last_obs_torch = preprocess_obs(obs).to(device)
            _, last_values = ac_net(last_obs_torch)

        returns, advantages = compute_gae(
            rewards_mix_buf,
            values_buf,
            dones_buf,
            last_values,
            gamma=config.gamma,
            lam=config.gae_lambda,
        )

        # Flatten
        b_obs = obs_buf.reshape(-1, *obs_shape)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

        # A2C update
        logits, values = ac_net(b_obs)
        dist = Categorical(logits=logits)
        new_logprobs = dist.log_prob(b_actions)
        entropy = dist.entropy().mean()

        policy_loss = -(b_advantages * new_logprobs).mean()
        value_loss = 0.5 * (b_returns - values).pow(2).mean()
        loss = policy_loss + config.value_coef * value_loss - config.ent_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ac_net.parameters(), config.max_grad_norm)
        optimizer.step()

        # Bandit update (AIRS only): use mean extrinsic episode return
        if config.mode == "airs":
            if len(ep_returns_this_update) > 0:
                mean_ret = float(np.mean(ep_returns_this_update))
                bandit.update(intrinsic_arm, mean_ret)

        # Basic logging
        if len(ep_returns_this_update) > 0:
            episode_returns.extend(ep_returns_this_update)
            mean_return = np.mean(episode_returns[-50:])
        else:
            mean_return = float("nan")

        if (update + 1) % 10 == 0:
            print(
                f"Update {update+1}/{num_updates} | "
                f"Mode {config.mode} | "
                f"Mean episode return (last 50) = {mean_return:.3f}"
            )

    if config.mode == "airs":
        print("AIRS arm selection counts:", arm_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_id",
        type=str,
        default="MiniGrid-Empty-16x16-v0",
        help="MiniGrid env id, e.g. MiniGrid-Empty-16x16-v0 or MiniGrid-DoorKey-6x6-v0",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["a2c", "a2c_re3", "airs"],
        default="a2c",
        help="Which method to run",
    )
    parser.add_argument(
        "--total_timesteps", type=int, default=200_000,
        help="Total env steps across all envs"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    cfg = TrainConfig(
        env_id=args.env_id,
        mode=args.mode,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
    )
    train(cfg)


if __name__ == "__main__":
    main()

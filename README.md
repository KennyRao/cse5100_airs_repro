<!-- README.md -->
# cse5100_airs_repro

Reproduction and extension of [**Automatic Intrinsic Reward Shaping (AIRS)**](https://arxiv.org/abs/2301.10886) on MiniGrid environments, with:

- plain A2C baseline
- A2C + RE3 intrinsic reward
- A2C + RISE intrinsic reward (Rényi state entropy)
- A2C + AIRS bandit over ID / RE3 / RISE
- **Cost-aware AIRS** extension that penalizes expensive intrinsic rewards in the bandit

---

## 1. Environment setup

Tested with:

- **Python:** 3.10
- **OS:** Windows
- **GPU:** optional but recommended (PyTorch with CUDA 12.1 in `requirements.txt`)

### 1.1 Clone this repository

```bash
git clone https://github.com/KennyRao/cse5100_airs_repro cse5100_airs_repro
cd cse5100_airs_repro
```

### 1.2 Create and activate a virtual environment
From inside the project folder:

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell or CMD)**

  ```bash
  .venv\Scripts\Activate
  ```

- **macOS / Linux**

  ```bash
  source .venv/bin/activate
  ```

You should now see `(.venv)` in your terminal prompt.

---

### 1.3 Install dependencies

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

---

## 2. Project Structure (Key Files)

- `airs/bandit.py`
  - `UCBIntrinsicBandit` with optional **cost_penalty** and **arm_costs**.
- `airs/intrinsic.py`
  - Intrinsic rewards:
    - `IdentityIntrinsic` (ID)
    - `RE3Intrinsic` (RE3)
    - `RISEIntrinsic` (RISE)
  - `beta_schedule` for intrinsic scaling.
- `airs/networks.py`
  - `ActorCriticNet` with:
    - Shared conv encoder
    - Policy head
    - Two value heads: task value and (task+intrinsic) value
  - `RandomEncoder` used by RE3/RISE.
- `scripts/train_minigrid.py`
  - Main training script for:
    - `a2c` (no intrinsic)
    - `a2c_re3` (A2C + RE3)
    - `a2c_rise` (A2C + RISE)
    - `airs` (bandit over {ID, RE3, RISE}, with optional cost penalty)
- `scripts/plot_minigrid_results.py`
  - Script to plot learning curves from CSV logs.
- `requirements.txt`
  - Python dependencies.

---

## 3. Hyperparameters and Command‑Line Flags

The `TrainConfig` dataclass contains all key hyperparameters. Important ones you may want to change via CLI:

- **Environment / mode**

  - `env_id`: e.g. `MiniGrid-Empty-16x16-v0` or `MiniGrid-DoorKey-6x6-v0`
  - `mode`: one of
    - `a2c`
    - `a2c_re3`
    - `a2c_rise`
    - `airs`

- **Total timesteps**

  - `total_timesteps`: default `1_000_000` (1M env steps across all envs)

- **AIRS cost penalty**

  - `airs_cost_penalty` (float): how strongly to penalize expensive intrinsic rewards.
    - `0.0` (default) → original AIRS (no cost‑awareness)
    - `> 0` → computation‑aware AIRS

---

## 4. Training Commands

All commands below assume:

- You are in the project root.
- Your virtual environment is activated.

### 4.1. Baselines

**Empty‑16x16**

```bash
# A2C without intrinsic reward
python -m scripts.train_minigrid   --env_id MiniGrid-Empty-16x16-v0   --mode a2c   --exp_name baseline

# A2C + RE3
python -m scripts.train_minigrid   --env_id MiniGrid-Empty-16x16-v0   --mode a2c_re3   --exp_name re3

# A2C + RISE
python -m scripts.train_minigrid   --env_id MiniGrid-Empty-16x16-v0   --mode a2c_rise   --exp_name rise
```

**DoorKey‑6x6**

```bash
# A2C without intrinsic reward
python -m scripts.train_minigrid   --env_id MiniGrid-DoorKey-6x6-v0   --mode a2c   --exp_name baseline

# A2C + RE3
python -m scripts.train_minigrid   --env_id MiniGrid-DoorKey-6x6-v0   --mode a2c_re3   --exp_name re3

# A2C + RISE
python -m scripts.train_minigrid   --env_id MiniGrid-DoorKey-6x6-v0   --mode a2c_rise   --exp_name rise
```

---

### 4.2. AIRS without cost penalty (original AIRS behavior)

**Empty‑16x16 (AIRS, λ = 0)**

```bash
python -m scripts.train_minigrid   --env_id MiniGrid-Empty-16x16-v0   --mode airs   --exp_name airs_cost0   --airs_cost_penalty 0.0
```

**DoorKey‑6x6 (AIRS, λ = 0)**

```bash
python -m scripts.train_minigrid   --env_id MiniGrid-DoorKey-6x6-v0   --mode airs   --exp_name airs_cost0   --airs_cost_penalty 0.0
```

These runs reproduce AIRS with **no computation cost penalty**.

---

### 4.3. AIRS with cost penalty (computation‑aware AIRS)

To compare, run a second experiment with `airs_cost_penalty > 0`.

**Empty‑16x16 (AIRS, λ > 0)**

```bash
python -m scripts.train_minigrid   --env_id MiniGrid-Empty-16x16-v0   --mode airs   --exp_name airs_cost1   --airs_cost_penalty 1.0
```

**DoorKey‑6x6 (AIRS, λ > 0)**

```bash
python -m scripts.train_minigrid   --env_id MiniGrid-DoorKey-6x6-v0   --mode airs   --exp_name airs_cost1   --airs_cost_penalty 1.0
```

This will train a **cost‑aware AIRS** variant that trades off reward against the compute cost of each intrinsic module.

---

## 5. Plotting Results

Once the training runs are done (they write CSV logs under `results/`), you can plot them with:

```bash
python -m scripts.plot_minigrid_results   --results_dir results   --out_path minigrid_comparison.png
```

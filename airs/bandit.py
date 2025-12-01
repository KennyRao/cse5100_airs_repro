# airs/bandit.py
import math
from collections import deque
from typing import Dict, List, Optional


class UCBIntrinsicBandit:
    """
    Simple UCB bandit over intrinsic reward choices.
    Arms are strings, e.g. ["id", "re3", "rise"].

    We keep a sliding window of recent mean extrinsic returns per arm
    and use UCB1:
        score(I) = Q(I) + c * sqrt(log T / N(I)) - cost_penalty * cost(I)
    """
    def __init__(self, arms: List[str], c: float = 1.0, window: int = 10, cost_penalty: float = 0.0, arm_costs: Optional[Dict[str, float]] = None):
        self.arms = arms
        self.c = c
        self.window = window
        self.cost_penalty = cost_penalty
        
        if arm_costs is None:
            # default: all arms have equal cost
            arm_costs = {arm: 1.0 for arm in arms}
        self.arm_costs: Dict[str, float] = dict(arm_costs)

        self.counts: Dict[str, int] = {arm: 1 for arm in arms}
        self.recent_returns: Dict[str, deque] = {
            arm: deque(maxlen=window) for arm in arms
        }
        self.total_updates = 0

        # For runtime measurement
        self.raw_cost_sum: Dict[str, float] = {arm: 0.0 for arm in arms}
        self.raw_cost_count: Dict[str, int] = {arm: 0 for arm in arms}

    def select_arm(self) -> str:
        self.total_updates += 1

        # Force each arm to be used at least once before UCB
        for arm in self.arms:
            if self.raw_cost_count[arm] == 0:
                return arm

        scores = {}
        for arm in self.arms:
            if len(self.recent_returns[arm]) == 0:
                q = 0.0
            else:
                q = sum(self.recent_returns[arm]) / len(self.recent_returns[arm])
            bonus = self.c * math.sqrt(
                math.log(self.total_updates + 1.0) / self.counts[arm]
            )
            cost_term = self.cost_penalty * self.arm_costs.get(arm, 0.0)
            scores[arm] = q + bonus - cost_term

        best = max(scores, key=scores.get)
        return best

    def update(self, arm: str, task_return_estimate: float):
        self.recent_returns[arm].append(float(task_return_estimate))
        self.counts[arm] += 1
    
    def record_cost(self, arm: str, elapsed_sec: float):
        """Record one timing sample for the given arm."""
        if arm not in self.raw_cost_sum:
            return
        self.raw_cost_sum[arm] += float(elapsed_sec)
        self.raw_cost_count[arm] += 1

    def recompute_arm_costs(self, base_arm: str = "id"):
        """
        Recompute arm_costs in a bounded, normalized way.
        """
        avg_times: Dict[str, float] = {}
        for arm in self.arms:
            cnt = self.raw_cost_count[arm]
            if cnt > 0:
                avg = self.raw_cost_sum[arm] / cnt
            else:
                # If we never timed this arm yet, pretend it is "normal"
                avg = 1.0
            avg_times[arm] = avg

        # Choose baseline
        if base_arm in avg_times and avg_times[base_arm] > 0.0:
            base = avg_times[base_arm]
        else:
            base = min(avg_times.values()) if len(avg_times) > 0 else 1.0
            if base <= 0.0:
                base = 1.0

        # 1) raw slowdown ratios (>= 0)
        raw_ratios: Dict[str, float] = {}
        for arm in self.arms:
            raw = avg_times[arm] / base
            raw_ratios[arm] = raw

        # 2) clip extreme ratios so we don't explode the scale
        R_MAX = 50.0  # e.g., treat anything > 50x as "very expensive but finite"
        for arm in self.arms:
            raw_ratios[arm] = min(raw_ratios[arm], R_MAX)

        # 3) normalize to [0, 1] so cheapest = 0, slowest = 1
        min_raw = min(raw_ratios.values())
        max_raw = max(raw_ratios.values())

        if max_raw == min_raw:
            # All arms look the same, no cost-based bias
            for arm in self.arms:
                self.arm_costs[arm] = 0.0
        else:
            for arm in self.arms:
                # cost 0 for fastest, 1 for slowest
                self.arm_costs[arm] = (raw_ratios[arm] - min_raw) / (max_raw - min_raw)

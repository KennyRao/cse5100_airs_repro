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
        scores = {}
        for arm in self.arms:
            if len(self.recent_returns[arm]) == 0:
                q = 0.0
            else:
                q = sum(self.recent_returns[arm]) / len(self.recent_returns[arm])
            bonus = self.c * math.sqrt(
                math.log(self.total_updates + 1.0) / self.counts[arm]
            )
            # Cost-aware UCB: penalize high-cost arms
            cost_term = self.cost_penalty * self.arm_costs.get(arm, 1.0)
            scores[arm] = q + bonus - cost_term

        # Greedy over UCB scores
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
        Recompute arm_costs as:
            cost[arm] = avg_time(arm) / avg_time(base_arm)
        so that cost[base_arm] ~ 1.0 and others are "times slower".
        If base_arm has no data yet, fall back to min avg time.
        """
        avg_times: Dict[str, float] = {}
        for arm in self.arms:
            cnt = self.raw_cost_count[arm]
            if cnt > 0:
                avg = self.raw_cost_sum[arm] / cnt
            else:
                avg = 1.0
            avg_times[arm] = avg
        
        # Choose baseline
        if base_arm in avg_times and avg_times[base_arm] > 0.0:
            base = avg_times[base_arm]
        else:
            # Smallest non-zero avg, or 1.0
            base = min(avg_times.values()) if len(avg_times) > 0 else 1.0
            if base <= 0.0:
                base = 1.0

        # Normalize
        for arm in self.arms:
            self.arm_costs[arm] = max(avg_times[arm] / base, 1e-3)

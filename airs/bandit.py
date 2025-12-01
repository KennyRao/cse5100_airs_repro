# airs/bandit.py
import math
from collections import deque
from typing import Dict, List


class UCBIntrinsicBandit:
    """
    Simple UCB bandit over intrinsic reward choices.
    Arms are strings, e.g. ["id", "re3", "rise"].

    We keep a sliding window of recent mean extrinsic returns per arm
    and use UCB1:
        score(I) = Q(I) + c * sqrt(log T / N(I))
    """
    def __init__(self, arms: List[str], c: float = 1.0, window: int = 10):
        self.arms = arms
        self.c = c
        self.window = window

        self.counts: Dict[str, int] = {arm: 1 for arm in arms}
        self.recent_returns: Dict[str, deque] = {
            arm: deque(maxlen=window) for arm in arms
        }
        self.total_updates = 0

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
            scores[arm] = q + bonus
        # Greedy over UCB scores
        best = max(scores, key=scores.get)
        return best

    def update(self, arm: str, task_return_estimate: float):
        self.recent_returns[arm].append(float(task_return_estimate))
        self.counts[arm] += 1

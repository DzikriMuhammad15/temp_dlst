from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RewardConfig:
    reward_mode: str = "delta_utility" 
    terminal_bonus: float = 1.0
    no_deal_penalty: float = -1.0
    time_penalty: float = 0.90
    utility_scale: bool = True


class RewardShaper:

    def __init__(self, cfg: RewardConfig, utility_min: float = 0.0, utility_max: float = 1.0):
        self.cfg = cfg
        self.utility_min = utility_min
        self.utility_max = utility_max

    def _scale(self, u: float) -> float:
        if not self.cfg.utility_scale:
            return float(u)
        denom = max(1e-8, self.utility_max - self.utility_min)
        return (float(u) - self.utility_min) / denom

    def compute_step_reward(
        self,
        step: Dict[str, Any],
        episode: Dict[str, Any],
        step_idx: int,
    ) -> float:
        u = step.utility

        t = step_idx

        if u is None:
            u = 0.0

        u = self._scale(float(u))

        mode = self.cfg.reward_mode.lower()
        r = 0.0

        if mode == "terminal_only":
            r = 0.0
        elif mode == "utility":
            r = u
        elif mode == "time_discounted_utility":
            r = u * (self.cfg.time_penalty ** t)
        else:
            raise ValueError(f"reward_mode tidak dikenal: {self.cfg.reward_mode}")

        is_last = step_idx == len(episode.steps) - 1
        if is_last:
            if episode.agreement is not None:
                r += self.cfg.terminal_bonus
            else:
                r += self.cfg.no_deal_penalty

        return float(r)

    def compute_reward_sequence(self, episode: Dict[str, Any]) -> List[float]:
        rewards = []
        steps = episode.steps
        for i, step in enumerate(steps):
            rewards.append(self.compute_step_reward(step, episode, i))
        return rewards
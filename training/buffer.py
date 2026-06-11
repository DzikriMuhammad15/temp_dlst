import json
import os
from typing import Any, Dict, List
from data_type.episode import Episode
from dataclasses import asdict, is_dataclass

import pandas as pd


class TrajectoryBuffer:
    """
    Buffer untuk menyimpan episode trajectory sebagai dataset statis.
    Format yang disimpan adalah JSONL:
    satu episode per baris.
    """

    def __init__(self):
        self.episodes: List[Episode] = []

    def add_episode(self, episode: Episode):
        self.episodes.append(episode)

    def extend(self, episodes: List[Episode]):
        self.episodes.extend(episodes)

    def clear(self):
        self.episodes = []

    def save_jsonl(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for episode in self.episodes:

                if is_dataclass(episode):
                    episode = asdict(episode)

                f.write(json.dumps(episode, ensure_ascii=False) + "\n")

    @staticmethod
    def load_jsonl(path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []

        episodes = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line:
                    episodes.append(json.loads(line))

        return episodes
    

    @staticmethod
    def episodes_to_dataframe(episodes: List[Any]) -> pd.DataFrame:
        rows = []

        for ep in episodes:

            if is_dataclass(ep):
                ep = asdict(ep)

            meta = ep.get("meta", {})

            for step_idx, step in enumerate(ep.get("steps", [])):

                row = {
                    "episode_id": ep.get("episode_id"),
                    "step_idx": step_idx,
                    "agent_name": step.get("agent_name"),
                    "kind": step.get("kind"),
                    "action": step.get("action"),
                    "utility": step.get("utility"),
                    "prev_utility": step.get("prev_utility"),
                    "relative_time": step.get("relative_time"),
                    "reward": ep.get("reward", 0.0),
                    "agreement": ep.get("agreement"),
                }

                row.update(meta)
                rows.append(row)

        return pd.DataFrame(rows)
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def plot_training_history(history: Dict[str, List[float]], save_dir: Optional[str] = None):
    """
    Setiap metrik dibuat figure terpisah agar mudah dibaca.
    """
    import os
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for key, values in history.items():
        if not values:
            continue

        plt.figure(figsize=(10, 4))
        plt.plot(values)
        plt.title(key)
        plt.xlabel("Episode")
        plt.ylabel(key)
        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{key}.png"), dpi=150, bbox_inches="tight")
        plt.show()


def plot_evaluation_metrics(results: Dict[str, List[float]], save_dir: Optional[str] = None):
    """
    Plot hasil evaluasi episode-by-episode.
    """
    import os
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for key, values in results.items():
        if not values:
            continue

        plt.figure(figsize=(10, 4))
        plt.plot(values)
        plt.title(key)
        plt.xlabel("Episode")
        plt.ylabel(key)
        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{key}.png"), dpi=150, bbox_inches="tight")
        plt.show()


def plot_agent_comparison_bar(labels, metric_values, title, save_path: Optional[str] = None):
    plt.figure(figsize=(10, 4))
    plt.bar(labels, metric_values)
    plt.title(title)
    plt.xlabel("Agent")
    plt.ylabel(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
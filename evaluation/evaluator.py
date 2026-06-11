from typing import Any, Callable, Dict, List, Tuple

from environments.negmas_env import NegMASBilateralEnv
from evaluation.metrics import aggregate_metrics, compute_episode_metrics


class BilateralEvaluator:
    """
    Evaluator untuk membandingkan dua agent atau banyak agent secara pairwise.
    """

    def __init__(self, domain_spec, max_steps: int = 20, device: str = "cpu"):
        self.domain_spec = domain_spec
        self.max_steps = max_steps
        self.device = device
        self.env = NegMASBilateralEnv(domain_spec=domain_spec, max_steps=max_steps)

    def run_pair(
        self,
        learner_factory: Callable[[], Any],
        opponent_factory: Callable[[], Any],
        episodes: int = 20,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        rows = []

        for _ in range(episodes):
            learner = learner_factory()
            opponent = opponent_factory()

            agreement, _ = self.env.run(learner, opponent)

            metrics = compute_episode_metrics(
                agreement=agreement,
                learner=learner,
                opponent=opponent,
                domain_spec=self.domain_spec,
                max_steps=self.max_steps,
            )
            rows.append(metrics)

        summary = aggregate_metrics(rows)
        return rows, summary

    def round_robin(
        self,
        agent_factories: Dict[str, Callable[[], Any]],
        episodes_per_pair: int = 10,
    ) -> Dict[str, Any]:
        names = list(agent_factories.keys())
        pairwise = {}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a_name = names[i]
                b_name = names[j]

                rows, summary = self.run_pair(
                    learner_factory=agent_factories[a_name],
                    opponent_factory=agent_factories[b_name],
                    episodes=episodes_per_pair,
                )
                pairwise[f"{a_name} vs {b_name}"] = {
                    "rows": rows,
                    "summary": summary,
                }

        return pairwise
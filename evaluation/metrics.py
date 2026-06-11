from types import SimpleNamespace
from typing import Any, Dict, List

from utils.utility import compute_utility


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(rows) == 0:
        return {}

    result = {}
    keys = rows[0].keys()
    for k in keys:
        try:
            vals = [float(r[k]) for r in rows]
            result[k] = sum(vals) / len(vals)
        except Exception:
            pass

    result["episodes"] = len(rows)
    result["deal_rate"] = sum(int(r["deal"]) for r in rows) / len(rows)
    return result


def compute_concession(agent_type, agreement, user_ufun=None, opponent_ufun=None, domain_spec=None, user_agent=None, opponent_agent=None):
    if agent_type == "learner":
        user_utils = []

        # Gunakan user_agent (bukan opponent_agent) dan ambil dari memory_propose
        episode_payload_learner = user_agent.build_episode_payload_propose(agreement=agreement)

        for step in episode_payload_learner.steps:
            utility = step.utility
            user_utils.append(utility)

        if len(user_utils) < 2:
            return 0.0

        return max(0.0, user_utils[0] - user_utils[-1])

    else:
        # type == "opponent"
        opponent_utils = []

        episode_payload_opponent = opponent_agent.build_episode_payload(agreement=agreement)

        for step in episode_payload_opponent.steps:
            utility = step.utility
            opponent_utils.append(utility)

        if len(opponent_utils) < 2:
            return 0.0
        return max(0.0, opponent_utils[0] - opponent_utils[-1])


def compute_episode_metrics(agreement, domain_spec, max_steps, learner_ufun, opponent_ufun, user_agent=None, opponent_agent=None) -> Dict[str, Any]:
    deal = agreement is not None

    if deal:
        u_learner = _safe_float(compute_utility(learner_ufun, agreement, domain_spec.issue_names))
        u_opponent = _safe_float(compute_utility(opponent_ufun, agreement, domain_spec.issue_names))
    else:
        u_learner = -1
        u_opponent = -1

    social_welfare = u_learner + u_opponent
    utility_gap = abs(u_learner - u_opponent)

    reserve_a = _safe_float(domain_spec.reserved_values.get("learner", 0.0))
    reserve_b = _safe_float(domain_spec.reserved_values.get("opponent", 0.0))
    nash_product = max(0.0, u_learner - reserve_a) * max(0.0, u_opponent - reserve_b)

    concession_learner = compute_concession("learner", agreement, learner_ufun, opponent_ufun, domain_spec, user_agent, opponent_agent)
    concession_opponent = compute_concession("opponent", agreement, learner_ufun, opponent_ufun, domain_spec, user_agent, opponent_agent)

    # Gunakan memory_propose untuk episode length (tiap propose = 1 langkah negosiasi)
    episode_payload_learner = user_agent.build_episode_payload_propose(
        agreement=agreement
    )

    return {
        "deal": int(deal),
        "u_learner": u_learner,
        "u_opponent": u_opponent,
        "social_welfare": social_welfare,
        "utility_gap": utility_gap,
        "nash_product": nash_product,
        "concession_learner": concession_learner,
        "concession_opponent": concession_opponent,
        "episode_length": len(episode_payload_learner.steps),
        "max_steps": max_steps,
    }
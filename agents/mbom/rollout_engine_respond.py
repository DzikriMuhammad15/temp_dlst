from typing import Callable, Dict, Any, List, Optional

import numpy as np

from environments.negmas_env import NegMASBilateralEnv

from agents.iop_agent import RandomIOPAgent

import random

class NegotiationRolloutEngineRespond:

    def __init__(
        self,
        domain_spec,
        action_library: List[Dict[str, Any]],
        issue_names: List[str],
        gamma: float = 0.99,
        rollout_k: int = 2,
    ):
        self.action_library = action_library
        self.issue_names = issue_names
        self.gamma = gamma
        self.rollout_k = rollout_k
        self.domain_spec = domain_spec

    def build_opponent_state(self, user_offer):
        return {"current_offer": user_offer}

    def compute_best_response(
            self,
            initial_user_state_basis,
            opponent_util_fn,
            opponent_value_fn,
            user_agent,
            relative_time,
            n_uniform_sample=5,
    ):
        best_action_idx = 0
        best_value = float("-inf")

        # Aksi user diambil dari agent ASLI (bukan clone)
        obs = user_agent.build_state(initial_user_state_basis)
        user_action_id, _, _ = user_agent.sample_action(obs, allow_accept=False)
        user_offer = user_agent.decode_action(user_action_id)
        opponent_state = self.build_opponent_state(user_offer)

        for candidate_idx in range(2): # hanya 2 respond action: accept (0) atau reject (1)
            for _ in range(n_uniform_sample):
                # print(f"Simulating rollout for candidate action idx: {candidate_idx}")

                # Clone BARU untuk setiap kandidat — bersih, tidak ada bocoran state
                agent_clone = user_agent.clone_for_rollout()

                rollout_value = self._simulate_rollout(
                    user_agent_clone=agent_clone,
                    opponent_state=opponent_state,
                    opponent_first_action=candidate_idx,
                    opponent_util_fn=opponent_util_fn,
                    oppponent_value_fn=opponent_value_fn,
                    original_ufun=user_agent.ufun,
                )

                if rollout_value > best_value:
                    best_value = rollout_value
                    best_action_idx = candidate_idx

        return best_action_idx

    def _simulate_rollout(
        self,
        user_agent_clone,
        opponent_state,
        opponent_first_action,
        opponent_util_fn,
        oppponent_value_fn=None,
        original_ufun=None,
    ) -> float:
        """
        Simulasikan satu rollout menggunakan clone agent (bukan agent asli).

        user_agent_clone : instance bersih dari clone_for_rollout()
        original_ufun    : ufun NegMAS yang valid dari agent asli — dipakai
                           oleh RandomIOPAgent agar kompatibel dengan NegMAS,
                           bukan lambda opponent_util_fn.
        """
        if self.rollout_k == 0:
            # print("rollout_k is 0, returning 0.0")
            return 0.0

        cummulative = 0
        current_offer_opponent_pov = opponent_state.get("current_offer")

        if opponent_first_action == 0:
            # Lawan langsung menerima offer user
            utility = float(opponent_util_fn(current_offer_opponent_pov))
            cummulative += utility
            return cummulative

        opponent_initial_offer = self.action_library[random.randint(0, len(self.action_library)-1)] # random offer untuk kasus reject, karena tidak ada offer baru dari opponent

        # RandomIOPAgent memakai original_ufun (NegMAS-compatible), bukan lambda
        ufun_for_opponent = original_ufun if original_ufun is not None else user_agent_clone.ufun
        opponent_agent = RandomIOPAgent(
            name="opponent_agent_rollout",
            ufun=ufun_for_opponent,
            domain_spec=self.domain_spec,
            state_dim=user_agent_clone.state_dim,
        )

        env = NegMASBilateralEnv(domain_spec=self.domain_spec, max_steps=10)

        # Clone sudah bersih — tidak perlu skip_agent_reset
        agreement, summary = env.run(
            user_agent_clone,
            opponent_agent,
            first_actor="a",
            start_with="respond",
            initial_offer=opponent_initial_offer,
            skip_agent_reset=False,
        )

        episode_payload = opponent_agent.build_episode_payload(agreement)

        t = 1
        k = 1
        for step in episode_payload.steps:
            if k == self.rollout_k:
                break
            reward = float(step.utility) * (self.gamma ** t)
            cummulative += reward
            t += 1
            k += 1

        if oppponent_value_fn is not None:
            value_estimate = float(oppponent_value_fn(current_offer_opponent_pov))
            cummulative += value_estimate * (self.gamma ** t)

        return cummulative
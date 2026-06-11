from copy import deepcopy
import numpy as np
import torch
from negmas import ResponseType
from agents.base_agent import BaseNegotiator
from utils.utility import compute_utility, object_to_dict
from data_type.step import Step
from data_type.episode import Episode


class RandomIOPAgent(BaseNegotiator):
    def __init__(
        self,
        name: str,
        ufun,
        domain_spec,
        state_dim: int = 3,
        trainers=None,
        **kwargs,
    ):
        super().__init__(name=name, ufun=ufun, domain_spec=domain_spec, trainers=trainers, **kwargs)

        self.action_library = self._domain_attr("action_library")
        self.propose_action_dim = len(self.action_library)  # propose: offer only
        self.respond_action_dim = 2                          # respond: 0=accept, 1=reject

        self.utility_min = self._domain_attr("utility_min")
        self.utility_max = self._domain_attr("utility_max")

        self.issue_names = self._domain_attr("issue_names")
        self.best_outcome = self._domain_attr("best_outcome")
        self.worst_outcome = self._domain_attr("worst_outcome")

        self.state_dim = state_dim

    # ──────────────────────────────────────────────────────────
    # SAMPLE ACTIONS
    # ──────────────────────────────────────────────────────────

    def sample_action_propose(self, obs=None):
        """
        Pilih propose action secara random: index 0-based ke action_library.
        """
        action_id = np.random.randint(0, self.propose_action_dim)
        log_prob = -np.log(self.propose_action_dim)
        return int(action_id), float(log_prob), None

    def sample_action_respond(self, obs=None):
        """
        Pilih respond action secara random: 0=accept, 1=reject.
        """
        action_id = np.random.randint(0, self.respond_action_dim)
        log_prob = -np.log(self.respond_action_dim)
        return int(action_id), float(log_prob), None

    # ──────────────────────────────────────────────────────────
    # Tidak butuh, override kosong
    # ──────────────────────────────────────────────────────────
    def attach_trainer(self, dict_of_trainer):
        pass

    def build_state(self, state, offer=None):
        pass

    def _mask_logits(self, logits, allow_accept):
        pass

    # ──────────────────────────────────────────────────────────
    # DECODE / ENCODE
    # ──────────────────────────────────────────────────────────
    def decode_action(self, action_id: int):
        idx = min(action_id, len(self.action_library) - 1)
        return self.action_library[idx]

    def _offer_to_action_idx(self, offer):
        if offer is None:
            return 0
        try:
            return self.action_library.index(offer)
        except ValueError:
            return 0

    def _normalize_utility(self, u: float) -> float:
        denom = max(1e-8, self.utility_max - self.utility_min)
        return (float(u) - self.utility_min) / denom

    def _offer_utility(self, offer):
        return compute_utility(self.ufun, offer, self.issue_names)

    def respond(self, state, source=None):
        offer = state.current_offer

        # Flush pending propose step
        if self.current_state_data_propose:
            next_state = object_to_dict(state)
            self.record_step_propose(
                state=self.current_state_data_propose["state"],
                action=self.current_state_data_propose["action"],
                utility=self.current_state_data_propose["utility"],
                value=self.current_state_data_propose["value"],
                log_prob=self.current_state_data_propose["log_prob"],
                next_state=next_state,
            )
            self.current_state_data_propose = None

        # Flush pending respond step
        if self.current_state_data_respond:
            next_state = object_to_dict(state)
            self.record_step_respond(
                state=self.current_state_data_respond["state"],
                action=self.current_state_data_respond["action"],
                utility=self.current_state_data_respond["utility"],
                value=self.current_state_data_respond["value"],
                log_prob=self.current_state_data_respond["log_prob"],
                next_state=next_state,
            )
            self.current_state_data_respond = None

        if offer is None:
            return ResponseType.REJECT_OFFER

        action_id, log_prob, value = self.sample_action_respond(obs=None)
        # action_id: 0=accept, 1=reject

        current_u = self._offer_utility(offer)

        self.current_state_data_respond = {
            "state": object_to_dict(state),
            "action": action_id,
            "utility": current_u,
            "value": value,
            "log_prob": log_prob,
        }

        if action_id == 0:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state, dest=None):
        offer_opponent = state.current_offer
        action_id, log_prob, value = self.sample_action_propose(obs=None)
        # action_id: 0-based index ke action_library

        offer = self.decode_action(action_id)
        current_u = self._offer_utility(offer)

        self.current_state_data_propose = {
            "state": object_to_dict(state),
            "action": action_id,
            "utility": current_u,
            "value": value,
            "log_prob": log_prob,
        }
        return offer

    def build_episode_payload(self, agreement):
        return self.build_episode_payload_propose(agreement)

    def build_episode_payload_propose(self, agreement):
        steps = deepcopy(self.memory_propose)
        states, actions, log_probs, values, utilities, next_states = [], [], [], [], [], []

        for step in steps:
            states.append(step.state)
            actions.append(int(step.action))
            log_probs.append(0.0 if step.log_prob is None else float(step.log_prob))
            values.append(0.0 if step.value is None else float(step.value))
            utilities.append(0.0 if step.utility is None else float(step.utility))
            next_states.append(step.next_state)

        return Episode(
            agent_name=self.name,
            terminal_utility=self._offer_utility(agreement) if agreement is not None else -1.0,
            agreement=agreement,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            utilities=utilities,
            q_values=None,
            steps=steps,
        )

    def build_episode_payload_respond(self, agreement):
        steps = deepcopy(self.memory_respond)
        states, actions, log_probs, values, utilities, next_states = [], [], [], [], [], []

        for step in steps:
            states.append(step.state)
            actions.append(int(step.action))
            log_probs.append(0.0 if step.log_prob is None else float(step.log_prob))
            values.append(0.0 if step.value is None else float(step.value))
            utilities.append(0.0 if step.utility is None else float(step.utility))
            next_states.append(step.next_state)

        return Episode(
            agent_name=self.name,
            terminal_utility=self._offer_utility(agreement) if agreement is not None else -1.0,
            agreement=agreement,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            utilities=utilities,
            q_values=None,
            steps=steps,
        )

    def on_negotiation_end(self, state):
        # Flush pending propose step
        if self.current_state_data_propose:
            next_state = object_to_dict(state)
            self.record_step_propose(
                state=self.current_state_data_propose["state"],
                action=self.current_state_data_propose["action"],
                utility=self.current_state_data_propose["utility"],
                value=self.current_state_data_propose["value"],
                log_prob=self.current_state_data_propose["log_prob"],
                next_state=next_state,
            )
            self.current_state_data_propose = None

        # Flush pending respond step
        if self.current_state_data_respond:
            next_state = object_to_dict(state)
            self.record_step_respond(
                state=self.current_state_data_respond["state"],
                action=self.current_state_data_respond["action"],
                utility=self.current_state_data_respond["utility"],
                value=self.current_state_data_respond["value"],
                log_prob=self.current_state_data_respond["log_prob"],
                next_state=next_state,
            )
            self.current_state_data_respond = None

        agreement = getattr(state, "agreement", None)

    def reset_episode(self):
        self.memory_propose = []
        self.memory_respond = []
        self.current_state_data_propose = None
        self.current_state_data_respond = None

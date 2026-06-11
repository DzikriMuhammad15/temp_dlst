from negmas import ResponseType

from agents.base_agent import BaseNegotiator
from utils.utility import compute_utility
from utils.utility import object_to_dict

from data_type.step import Step
from data_type.episode import Episode

from copy import deepcopy


class RuleBasedAgent(BaseNegotiator):
    def __init__(self, name, ufun, domain_spec, trainers=None, initial_threshold=0.6, concession_power=1.5, **kwargs):
        super().__init__(name=name, ufun=ufun, domain_spec=domain_spec, trainers=trainers, **kwargs)

        self.action_library = self._domain_attr("action_library")
        self.utility_min = float(self._domain_attr("utility_min"))
        self.utility_max = float(self._domain_attr("utility_max"))
        self.issue_names = self._domain_attr("issue_names")

        self.initial_threshold = initial_threshold
        self.concession_power = concession_power
        # Dipisah sesuai dengan base_agent baru
        self.current_state_data_propose = None
        self.current_state_data_respond = None

    def _normalize(self, u):
        return (u - self.utility_min) / max(1e-8, self.utility_max - self.utility_min)

    def _denormalize(self, unorm):
        return self.utility_min + unorm * (self.utility_max - self.utility_min)

    def _aspiration(self, t):
        t = min(max(float(t), 0.0), 1.0)
        return 1.0 - (t ** self.concession_power) * (1.0 - self.initial_threshold)

    def _closest_offer(self, target):
        best, best_gap = None, float("inf")
        for o in self.action_library:
            u = compute_utility(self.ufun, o, self.issue_names)
            gap = abs(u - target)
            if gap < best_gap:
                best_gap = gap
                best = o
        return best if best is not None else self.action_library[0]

    def attach_trainer(self, dict_of_trainer):
        pass

    def build_state(self, state, offer=None):
        pass

    def reset_episode(self):
        self.memory_propose = []
        self.memory_respond = []
        self.current_state_data_propose = None
        self.current_state_data_respond = None

    def build_episode_payload(self, agreement):
        """
        Backward compat — mengembalikan propose payload.
        """
        return self.build_episode_payload_propose(agreement)

    def build_episode_payload_propose(self, agreement):
        """
        Format episode propose untuk dataset statis.
        """
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
        """
        Format episode respond untuk dataset statis.
        """
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

    def _mask_logits(self, logits, allow_accept: bool):
        if allow_accept:
            return logits
        masked = logits.clone()
        masked[..., 0] = -1e9
        return masked

    def sample_action_propose(self, obs):
        pass

    def sample_action_respond(self, obs):
        pass

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

    def _offer_utility(self, offer):
        return compute_utility(self.ufun, offer, self.issue_names)

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

        if len(self.trainers) > 0:
            payload_propose = self.build_episode_payload_propose(agreement)
            payload_respond = self.build_episode_payload_respond(agreement)
            for trainer in self.trainers.values():
                if isinstance(trainer, list):
                    continue
                # rule agent tidak punya propose/respond trainer secara default
                # tapi jika ada, simpan
                if hasattr(trainer, 'store_episode'):
                    trainer.store_episode(payload_propose)

    def respond(self, state, source=None):
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

        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        u = compute_utility(self.ufun, offer, self.issue_names)
        u_norm = self._normalize(u)
        target = self._aspiration(state.relative_time)

        current_u = self._offer_utility(offer)

        if u_norm >= target:
            # accept action = 0
            self.current_state_data_respond = {
                "state": object_to_dict(state),
                "action": 0,  # accept
                "utility": current_u,
                "value": None,
                "log_prob": None,
            }
            return ResponseType.ACCEPT_OFFER

        # reject action = 1
        self.current_state_data_respond = {
            "state": object_to_dict(state),
            "action": 1,  # reject
            "utility": current_u,
            "value": None,
            "log_prob": None,
        }
        return ResponseType.REJECT_OFFER

    def propose(self, state, dest=None):
        target = self._aspiration(state.relative_time)
        target_u = self._denormalize(target)

        offer = self._closest_offer(target_u)
        utility = compute_utility(self.ufun, offer, self.issue_names)
        action_id = self._offer_to_action_idx(offer)

        self.current_state_data_propose = {
            "state": object_to_dict(state),
            "action": action_id,
            "utility": utility,
            "value": None,
            "log_prob": None,
        }
        return offer

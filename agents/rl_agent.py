from copy import deepcopy

import numpy as np
import torch

from negmas import ResponseType

from agents.base_agent import BaseNegotiator
from models.policy_network import PolicyNetwork
from models.value_network import ValueNetwork
from utils.utility import compute_utility
from utils.utility import object_to_dict

from data_type.step import Step
from data_type.episode import Episode

class RLNegotiator(BaseNegotiator):

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
        # propose action dim = len(action_library) (tidak ada ACCEPT di propose)
        self.propose_action_dim = len(self.action_library)
        # respond action dim = 2 (0=accept, 1=reject)
        self.respond_action_dim = 2

        self.utility_min = self._domain_attr("utility_min")
        self.utility_max = self._domain_attr("utility_max")

        self.issue_names = self._domain_attr("issue_names")
        self.best_outcome = self._domain_attr("best_outcome")
        self.worst_outcome = self._domain_attr("worst_outcome")

        self.state_dim = state_dim
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"

    def attach_trainer(self, dict_of_trainer):
        """
        harus: agent.attach_trainer({
            "propose_trainer": PPOTrainer/PolicyGradientTrainer,
            "respond_trainer": PPOTrainer/PolicyGradientTrainer,
        })
        """
        for key, trainer in dict_of_trainer.items():
            self.trainers[key] = trainer

    def build_state(self, state, offer=None) -> np.ndarray:
        """
        State basis (dim = state_dim = 3): [current_u_norm, relative_time, has_offer]

        Menerima tiga bentuk input untuk parameter `state`:
        1. np.ndarray / list  — state vector yang sudah diproses (pass-through langsung).
        2. dict               — hasil object_to_dict(SAOState), dibaca via .get().
        3. SAOState object    — objek NegMAS asli, dibaca via getattr().

        Parameter `offer` bersifat opsional:
        - Jika diisi eksplisit → digunakan langsung.
        - Jika None + state adalah dict/object → ambil dari state.current_offer.
        - Jika None + current_offer juga None → fallback ke self.best_outcome.
        """
        # ── Cabang 1: sudah berupa state vector ─────────────────────────
        if isinstance(state, (np.ndarray, list)):
            arr = np.asarray(state, dtype=np.float32)
            if arr.shape == (self.state_dim,):
                return arr

        # ── Cabang 2: dict (dari object_to_dict) ────────────────────────
        if isinstance(state, dict):
            relative_time = float(state.get("relative_time", 0.0))
            if offer is None:
                offer = state.get("current_offer", None) or self.best_outcome

        # ── Cabang 3: SAOState object (NegMAS asli) ──────────────────────
        else:
            relative_time = float(getattr(state, "relative_time", 0.0))
            if offer is None:
                offer = getattr(state, "current_offer", None) or self.best_outcome

        # ── Hitung fitur ────────────────────────────────────────────────
        current_u    = self._offer_utility(offer)
        current_norm = self._normalize_utility(current_u)
        has_offer    = 1.0 if offer is not None else 0.0

        return np.array([current_norm, relative_time, has_offer], dtype=np.float32)

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

    # Backward compat: build_episode_payload mengembalikan propose payload
    def build_episode_payload(self, agreement):
        return self.build_episode_payload_propose(agreement)

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

            propose_trainer = self.trainers.get("propose_trainer")
            respond_trainer = self.trainers.get("respond_trainer")


            if propose_trainer is not None:
                propose_trainer.store_episode(payload_propose)
            if respond_trainer is not None:
                respond_trainer.store_episode(payload_respond)


    def _mask_logits(self, logits, allow_accept: bool):
        if allow_accept:
            return logits
        masked = logits.clone()
        masked[..., 0] = -1e9
        return masked

    def sample_action_propose(self, obs):
        """
        Sample aksi dari propose policy network.
        Aksi: 0..len(action_library)-1 (index langsung ke action_library).
        """
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        propose_trainer = self.trainers.get("propose_trainer")
        if propose_trainer is None:
            raise ValueError(
                "No propose_trainer attached to the agent. "
                "Please attach a trainer using agent.attach_trainer({'propose_trainer': your_trainer})"
            )

        policy = propose_trainer.models.get("policy").get("model")
        value_entry = propose_trainer.models.get("value")

        value = None
        with torch.no_grad():
            logits = policy(x)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if value_entry is not None:
                value_net = value_entry.get("model")
                value = value_net(x)

        return int(action.item()), float(log_prob.item()), float(value.item()) if value is not None else None

    def sample_action_respond(self, obs):
        """
        Sample aksi dari respond policy network.
        Aksi: 0=accept, 1=reject.
        """
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        respond_trainer = self.trainers.get("respond_trainer")
        if respond_trainer is None:
            raise ValueError(
                "No respond_trainer attached to the agent. "
                "Please attach a trainer using agent.attach_trainer({'respond_trainer': your_trainer})"
            )

        policy = respond_trainer.models.get("policy").get("model")
        value_entry = respond_trainer.models.get("value")

        value = None
        with torch.no_grad():
            logits = policy(x)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if value_entry is not None:
                value_net = value_entry.get("model")
                value = value_net(x)

        return int(action.item()), float(log_prob.item()), float(value.item()) if value is not None else None

    # Backward compat
    def sample_action(self, obs, allow_accept=True):
        if allow_accept:
            return self.sample_action_respond(obs)
        else:
            action_id, log_prob, value = self.sample_action_propose(obs)
            # Propose action_id adalah 0-based index ke action_library
            # Kembalikan sebagai action_id+1 agar decode_action (lama) masih bekerja
            return action_id + 1, log_prob, value

    def decode_action(self, action_id: int):
        """
        Decode action_id dari propose: action_id adalah index 0-based ke action_library.
        """
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
        print(f"respond() called with offer: {offer}")

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

        obs = self.build_state(state, offer)
        action_id, log_prob, value = self.sample_action_respond(obs)
        # action_id: 0=accept, 1=reject

        current_u = self._offer_utility(offer)

        # Simpan current respond state data (akan di-flush di respond() berikutnya atau on_negotiation_end())
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
        print(f"propose() called with opponent offer: {offer_opponent}")
        obs = self.build_state(state, offer_opponent)
        action_id, log_prob, value = self.sample_action_propose(obs)
        # action_id: 0-based index ke action_library

        offer = self.decode_action(action_id)
        current_u = self._offer_utility(offer)

        # Simpan current propose state data (akan di-flush di respond() berikutnya atau on_negotiation_end())
        self.current_state_data_propose = {
            "state": object_to_dict(state),
            "action": action_id,
            "utility": current_u,
            "value": value,
            "log_prob": log_prob,
        }
        return offer

    def reset_episode(self):
        self.memory_propose = []
        self.memory_respond = []
        self.current_state_data_propose = None
        self.current_state_data_respond = None

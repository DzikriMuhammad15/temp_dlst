from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
from agents.mbom.rollout_engine_propose import NegotiationRolloutEnginePropose
from agents.mbom.rollout_engine_respond import NegotiationRolloutEngineRespond
import torch

from negmas import ResponseType

from agents.base_agent import BaseNegotiator
from agents.mbom.iop_model import IOPNetwork
from agents.mbom.bayesian_mixer import BayesianMixer
from utils.utility import compute_utility
from utils.utility import object_to_dict

from data_type.step import Step
from data_type.episode import Episode


class MBOMAgent(BaseNegotiator):

    def __init__(
        self,
        name: str,
        ufun,
        domain_spec,
        state_dim: int = 3,
        M: int = 3,
        rollout_k: int = 2,
        gamma: float = 0.99,
        iop_batch_size: int = 32,
        iop_update_steps: int = 3,
        bayesian_decay: float = 0.9,
        bayesian_horizon: int = 10,
        bayesian_temperature: float = 1.0,
        n_rollout_samples: int = 5,
        trainers=None,
        **kwargs,
    ):
        super().__init__(
            name=name, ufun=ufun, domain_spec=domain_spec,
            trainers=trainers, **kwargs
        )

        # ── Domain info ────────────────────────────────────────────
        self.action_library: List[Dict] = self._domain_attr("action_library")
        self.issue_names: List[str] = self._domain_attr("issue_names")
        self.utility_min    = self._domain_attr("utility_min")
        self.utility_max    = self._domain_attr("utility_max")
        self.best_outcome   = self._domain_attr("best_outcome")
        self.worst_outcome  = self._domain_attr("worst_outcome")

        # ── Action dimensions ────────────────────────────────────────
        # propose action dim = len(action_library) (tidak ada ACCEPT)
        self.propose_action_dim = len(self.action_library)
        # respond action dim = 2 (0=accept, 1=reject)
        self.respond_action_dim = 2

        # ── State dimensions ─────────────────────────────────────────
        # state_dim          = dimensi state basis (default 3)
        # iop_propose_dim    = output IOP propose network = len(action_library)
        # iop_respond_dim    = output IOP respond network = 2
        # state_dim_augmented = state_dim + iop_respond_dim + iop_propose_dim
        self.state_dim             = state_dim
        self.iop_propose_action_dim = len(self.action_library)   # offer only
        self.iop_respond_action_dim = 2                           # 0=accept, 1=reject
        self.state_dim_augmented   = state_dim + self.iop_respond_action_dim + self.iop_propose_action_dim

        # ── MBOM hyperparameters ──────────────────────────────────────
        self.M               = M
        self.rollout_k       = rollout_k
        self.gamma           = gamma
        self.iop_batch_size  = iop_batch_size
        self.iop_update_steps = iop_update_steps
        self.n_rollout_samples = n_rollout_samples

        # ── Bayesian Mixers (satu untuk propose, satu untuk respond) ──
        self.bayesian_mixer_propose = BayesianMixer(
            M=M,
            decay=bayesian_decay,
            horizon=bayesian_horizon,
            temperature=bayesian_temperature,
        )
        self.bayesian_mixer_respond = BayesianMixer(
            M=M,
            decay=bayesian_decay,
            horizon=bayesian_horizon,
            temperature=bayesian_temperature,
        )

        # ── Rollout Engine ────────────────────────────────────────────
        self.rollout_engine_propose = NegotiationRolloutEnginePropose(
            action_library=self.action_library,
            issue_names=self.issue_names,
            gamma=gamma,
            rollout_k=rollout_k,
            domain_spec=self.domain_spec
        )

        self.rollout_engine_respond = NegotiationRolloutEngineRespond(
            action_library=self.action_library,
            issue_names=self.issue_names,
            gamma=gamma,
            rollout_k=rollout_k,
            domain_spec=self.domain_spec
        )

        # ── Device ────────────────────────────────────────────────────
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"

        # ── Per-episode state tracking ────────────────────────────────
        self.current_state_data_propose = None
        self.current_state_data_respond = None

        # rollout_level = None  → mode normal
        # rollout_level = int n → mode rollout (hanya IOP level-n)
        self.rollout_level = None

        # penyimpanan opponent state
        self.opponent_state = None

    # ──────────────────────────────────────────────────────────────────
    # TRAINER ATTACHMENT
    # ──────────────────────────────────────────────────────────────────

    def attach_trainer(self, dict_of_trainer):
        """
        harus: agent.attach_trainer({
            "propose_trainer": trainer,
            "respond_trainer": trainer,
            "iop_trainers_propose": IOPTrainer[],
            "iop_trainers_respond": IOPTrainer[],
        })
        """
        for key, trainer in dict_of_trainer.items():
            self.trainers[key] = trainer

        iop_propose = dict_of_trainer.get("iop_trainers_propose")
        if iop_propose is not None:
            self.M = len(iop_propose)

    # ──────────────────────────────────────────────────────────────────
    # STATE BUILDING
    # ──────────────────────────────────────────────────────────────────

    def build_state_basis(self, state, offer=None) -> np.ndarray:
        """
        State basis (dim = state_dim = 3): [current_u_norm, time, has_offer]
        """
        if isinstance(state, np.ndarray):
            return state.astype(np.float32)

        if isinstance(state, dict):
            relative_time = float(state.get("relative_time", 0.0))
            if offer is None:
                raw_offer = state.get("current_offer", None)
                offer = raw_offer if raw_offer is not None else self.best_outcome
        else:
            relative_time = float(getattr(state, "relative_time", 0.0))
            if offer is None:
                raw_offer = getattr(state, "current_offer", None)
                offer = raw_offer if raw_offer is not None else self.best_outcome

        current_u    = self._offer_utility(offer)
        current_norm = self._normalize_utility(current_u)
        has_offer    = 1.0 if offer is not None else 0.0

        return np.array([current_norm, relative_time, has_offer], dtype=np.float32)

    def build_state(self, state, offer=None) -> np.ndarray:
        """
        Public interface — mengembalikan AUGMENTED state (state_dim_augmented,).
        state_dim_augmented = state_dim + iop_respond_dim + iop_propose_dim

        Dipanggil oleh trainer._extract_states_actions() sehingga states
        yang direkonstruksi dari episode sudah berukuran state_dim_augmented
        dan langsung bisa dimasukkan ke policy network.
        """
        if isinstance(state, np.ndarray) and state.shape[0] == self.state_dim_augmented:
            return state.astype(np.float32)

        state_basis = self.build_state_basis(state, offer)
        return self._build_augmented_state(state_basis)

    # ──────────────────────────────────────────────────────────────────
    # AUGMENTED STATE
    # ──────────────────────────────────────────────────────────────────

    def _build_augmented_state(self, state_basis: np.ndarray) -> np.ndarray:
        """
        Augmentasi state basis dengan distribusi IOP respond dan IOP propose.

        Format: [state_basis | iop_respond_probs | iop_propose_probs]
        Dimensi: state_dim + iop_respond_dim + iop_propose_dim = state_dim_augmented

        rollout_level = None: bayesian mixing semua M model
        rollout_level = int n: hanya IOP level-n
        """
        iop_trainers_propose = self.trainers.get("iop_trainers_propose")
        iop_trainers_respond = self.trainers.get("iop_trainers_respond")

        x = torch.tensor(state_basis, dtype=torch.float32,
                         device=self.device).unsqueeze(0)

        # ── Hitung IOP respond probs ─────────────────────────────────
        if self.rollout_level is None:
            if iop_trainers_respond is None:
                uniform_resp = np.ones(self.iop_respond_action_dim, dtype=np.float32) / self.iop_respond_action_dim
            else:
                all_resp_probs: List[np.ndarray] = []
                for m in range(self.M):
                    iop_entry = iop_trainers_respond[m].models.get("policy")
                    if iop_entry is None:
                        uniform_resp_m = np.ones(self.iop_respond_action_dim, dtype=np.float32) / self.iop_respond_action_dim
                        all_resp_probs.append(uniform_resp_m)
                        continue
                    iop_net = iop_entry["model"]
                    iop_net.eval()
                    with torch.no_grad():
                        logits = iop_net(x)
                        probs  = torch.softmax(logits, dim=-1)
                    all_resp_probs.append(probs.cpu().numpy().flatten())
                uniform_resp = self.bayesian_mixer_respond.get_mixed_probs(all_resp_probs)
        else:
            n = self.rollout_level
            if iop_trainers_respond is None:
                uniform_resp = np.ones(self.iop_respond_action_dim, dtype=np.float32) / self.iop_respond_action_dim
            else:
                iop_entry = iop_trainers_respond[n].models.get("policy")
                if iop_entry is None:
                    uniform_resp = np.ones(self.iop_respond_action_dim, dtype=np.float32) / self.iop_respond_action_dim
                else:
                    iop_net = iop_entry["model"]
                    iop_net.eval()
                    with torch.no_grad():
                        logits = iop_net(x)
                        probs  = torch.softmax(logits, dim=-1)
                    uniform_resp = probs.cpu().numpy().flatten()

        iop_respond_probs = uniform_resp

        # ── Hitung IOP propose probs ─────────────────────────────────
        if self.rollout_level is None:
            if iop_trainers_propose is None:
                uniform_prop = np.ones(self.iop_propose_action_dim, dtype=np.float32) / self.iop_propose_action_dim
            else:
                all_prop_probs: List[np.ndarray] = []
                for m in range(self.M):
                    iop_entry = iop_trainers_propose[m].models.get("policy")
                    if iop_entry is None:
                        uniform_prop_m = np.ones(self.iop_propose_action_dim, dtype=np.float32) / self.iop_propose_action_dim
                        all_prop_probs.append(uniform_prop_m)
                        continue
                    iop_net = iop_entry["model"]
                    iop_net.eval()
                    with torch.no_grad():
                        logits = iop_net(x)
                        probs  = torch.softmax(logits, dim=-1)
                    all_prop_probs.append(probs.cpu().numpy().flatten())
                uniform_prop = self.bayesian_mixer_propose.get_mixed_probs(all_prop_probs)
        else:
            n = self.rollout_level
            if iop_trainers_propose is None:
                uniform_prop = np.ones(self.iop_propose_action_dim, dtype=np.float32) / self.iop_propose_action_dim
            else:
                iop_entry = iop_trainers_propose[n].models.get("policy")
                if iop_entry is None:
                    uniform_prop = np.ones(self.iop_propose_action_dim, dtype=np.float32) / self.iop_propose_action_dim
                else:
                    iop_net = iop_entry["model"]
                    iop_net.eval()
                    with torch.no_grad():
                        logits = iop_net(x)
                        probs  = torch.softmax(logits, dim=-1)
                    uniform_prop = probs.cpu().numpy().flatten()

        iop_propose_probs = uniform_prop

        # ── Gabungkan: [state_basis | iop_respond_probs | iop_propose_probs] ─
        return np.concatenate([state_basis, iop_respond_probs, iop_propose_probs], axis=0).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────
    # SAMPLE ACTION
    # ──────────────────────────────────────────────────────────────────

    def sample_action_propose(self, obs):
        """
        Sample aksi dari propose policy network.
        Input obs: augmented state (state_dim_augmented,)
        Output: (action_id, log_prob, value)
          action_id: 0-based index ke action_library (0..len(action_library)-1)
        """
        x = torch.tensor(obs, dtype=torch.float32,
                         device=self.device).unsqueeze(0)

        propose_trainer = self.trainers.get("propose_trainer")
        if propose_trainer is None:
            raise ValueError(
                "No propose_trainer attached to the agent. "
                "Please attach a trainer using "
                "agent.attach_trainer({'propose_trainer': your_trainer})"
            )

        policy_entry = propose_trainer.models.get("policy")
        if policy_entry is None:
            raise ValueError("No 'policy' model found in propose_trainer.models.")
        policy = policy_entry["model"]

        value_entry = propose_trainer.models.get("value")

        value = None
        with torch.no_grad():
            logits   = policy(x)
            dist     = torch.distributions.Categorical(logits=logits)
            action   = dist.sample()
            log_prob = dist.log_prob(action)
            if value_entry is not None:
                value_net = value_entry["model"]
                value     = value_net(x)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()) if value is not None else None,
        )

    def sample_action_respond(self, obs):
        """
        Sample aksi dari respond policy network.
        Input obs: augmented state (state_dim_augmented,)
        Output: (action_id, log_prob, value)
          action_id: 0=accept, 1=reject
        """
        x = torch.tensor(obs, dtype=torch.float32,
                         device=self.device).unsqueeze(0)

        respond_trainer = self.trainers.get("respond_trainer")
        if respond_trainer is None:
            raise ValueError(
                "No respond_trainer attached to the agent. "
                "Please attach a trainer using "
                "agent.attach_trainer({'respond_trainer': your_trainer})"
            )

        policy_entry = respond_trainer.models.get("policy")
        if policy_entry is None:
            raise ValueError("No 'policy' model found in respond_trainer.models.")
        policy = policy_entry["model"]

        value_entry = respond_trainer.models.get("value")

        value = None
        with torch.no_grad():
            logits   = policy(x)
            dist     = torch.distributions.Categorical(logits=logits)
            action   = dist.sample()
            log_prob = dist.log_prob(action)
            if value_entry is not None:
                value_net = value_entry["model"]
                value     = value_net(x)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()) if value is not None else None,
        )

    # Backward compat helper
    def sample_action(self, obs, allow_accept=True):
        if allow_accept:
            return self.sample_action_respond(obs)
        else:
            action_id, log_prob, value = self.sample_action_propose(obs)
            return action_id + 1, log_prob, value

    # ──────────────────────────────────────────────────────────────────
    # DECODE / ENCODE ACTION
    # ──────────────────────────────────────────────────────────────────

    def decode_action(self, action_id: int):
        """action_id adalah 0-based index ke action_library."""
        idx = min(action_id, len(self.action_library) - 1)
        return self.action_library[idx]

    def _offer_to_action_idx(self, offer) -> int:
        if offer is None:
            return 0
        try:
            return self.action_library.index(offer)
        except ValueError:
            return 0

    # ──────────────────────────────────────────────────────────────────
    # UTILITY HELPERS
    # ──────────────────────────────────────────────────────────────────

    def _normalize_utility(self, u: float) -> float:
        denom = max(1e-8, self.utility_max - self.utility_min)
        return (float(u) - self.utility_min) / denom

    def _offer_utility(self, offer) -> float:
        return compute_utility(self.ufun, offer, self.issue_names)

    def _mask_logits(self, logits, allow_accept: bool):
        if allow_accept:
            return logits
        masked = logits.clone()
        masked[..., 0] = -1e9
        return masked

    # ──────────────────────────────────────────────────────────────────
    # IOP OFFER/RESPOND INDEX HELPERS
    # ──────────────────────────────────────────────────────────────────

    def _iop_offer_to_propose_idx(self, offer) -> int:
        """
        Ubah offer ke indeks di IOP propose action space (0-based, sama dengan action_library index).
        """
        if offer is None:
            return 0
        try:
            return self.action_library.index(offer)
        except ValueError:
            u_target  = self._offer_utility(offer)
            best_idx, best_diff = 0, float("inf")
            for idx, lib_offer in enumerate(self.action_library):
                diff = abs(self._offer_utility(lib_offer) - u_target)
                if diff < best_diff:
                    best_diff = diff
                    best_idx  = idx
            return best_idx

    # ──────────────────────────────────────────────────────────────────
    # OPPONENT MODELING PIPELINE — PROPOSE
    # ──────────────────────────────────────────────────────────────────

    def _run_recursive_imagination_propose(
        self, state_basis: np.ndarray, relative_time: float
    ) -> None:
        """
        Recursive Imagination untuk IOP propose.
        """
        iop_trainers_propose = self.trainers.get("iop_trainers_propose")
        if iop_trainers_propose is None:
            return

        for m in range(1, self.M):
            self.rollout_level = m - 1
            iop_trainers_propose[m].copy_iop_policy_weight(
                iop_trainers_propose[m - 1].models.get("policy").get("model")
            )

            prev_m = m - 1
            def iop_prev_action_fn(sv: np.ndarray, _m=prev_m) -> int:
                iop_policy_net = iop_trainers_propose[_m].models.get("policy").get("model")
                if iop_policy_net is None:
                    return np.random.randint(0, self.iop_propose_action_dim)
                iop_policy_net.eval()
                x = torch.tensor(sv, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = iop_policy_net(x)
                    dist   = torch.distributions.Categorical(logits=logits)
                    return int(dist.sample().item())

            def opponent_value_fn(sv: np.ndarray) -> float:
                propose_trainer = self.trainers.get("propose_trainer")
                if propose_trainer is None:
                    return 0.0
                value_entry = propose_trainer.models.get("value")
                if value_entry is None:
                    return 0.0
                value_net = value_entry["model"]
                aug = self.build_state(sv)
                x   = torch.tensor(aug, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
                with torch.no_grad():
                    v = value_net(x)
                return -float(v.item())

            def opponent_utility_fn(offer) -> float:
                return -self._offer_utility(offer)

            print(f"Starting rollout for IOP propose level {m} with opponent POV...")
            best_response_idx = self.rollout_engine_propose.compute_best_response(
                initial_user_state_basis=state_basis,
                opponent_util_fn=opponent_utility_fn,
                opponent_value_fn=opponent_value_fn,
                user_agent=deepcopy(self),
                relative_time=relative_time,
                n_uniform_sample=1,
            )

            iop_trainers_propose[m].add_step(Step(
                state=state_basis,
                action=best_response_idx
            ))
            iop_trainers_propose[m].update()
            iop_trainers_propose[m].reset_step()

        self.rollout_level = None

    def _update_bayesian_mixer_propose(
        self, state_basis: np.ndarray, real_opponent_action_idx: int
    ) -> None:
        """Update Bayesian mixer untuk IOP propose."""
        iop_trainers_propose = self.trainers.get("iop_trainers_propose")
        if iop_trainers_propose is None:
            return

        iop_probs = []
        for m in range(self.M):
            iop_net = iop_trainers_propose[m].models.get("policy").get("model")
            if iop_net is None:
                iop_probs.append(1.0 / self.iop_propose_action_dim)
                continue
            iop_net.eval()
            x = torch.tensor(
                state_basis, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                logits = iop_net(x)
                probs  = torch.softmax(logits, dim=-1)
            p = float(probs[0, real_opponent_action_idx].item())
            iop_probs.append(p)

        self.bayesian_mixer_propose.update(np.array(iop_probs, dtype=np.float64))

    # ──────────────────────────────────────────────────────────────────
    # OPPONENT MODELING PIPELINE — RESPOND
    # ──────────────────────────────────────────────────────────────────

    def _run_recursive_imagination_respond(
        self, state_basis: np.ndarray, relative_time: float
    ) -> None:
        """
        Recursive Imagination untuk IOP respond.
        """
        iop_trainers_respond = self.trainers.get("iop_trainers_respond")
        if iop_trainers_respond is None:
            return

        for m in range(1, self.M):
            self.rollout_level = m - 1
            iop_trainers_respond[m].copy_iop_policy_weight(
                iop_trainers_respond[m - 1].models.get("policy").get("model")
            )

            prev_m = m - 1
            def iop_prev_action_fn_resp(sv: np.ndarray, _m=prev_m) -> int:
                iop_policy_net = iop_trainers_respond[_m].models.get("policy").get("model")
                if iop_policy_net is None:
                    return np.random.randint(0, self.iop_respond_action_dim)
                iop_policy_net.eval()
                x = torch.tensor(sv, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = iop_policy_net(x)
                    dist   = torch.distributions.Categorical(logits=logits)
                    return int(dist.sample().item())

            def opponent_value_fn_resp(sv: np.ndarray) -> float:
                respond_trainer = self.trainers.get("respond_trainer")
                if respond_trainer is None:
                    return 0.0
                value_entry = respond_trainer.models.get("value")
                if value_entry is None:
                    return 0.0
                value_net = value_entry["model"]
                aug = self.build_state(sv)
                x   = torch.tensor(aug, dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
                with torch.no_grad():
                    v = value_net(x)
                return -float(v.item())

            def opponent_utility_fn_resp(offer) -> float:
                return -self._offer_utility(offer)

            print(f"Starting rollout for IOP respond level {m} with opponent POV...")
            best_response_idx = self.rollout_engine_respond.compute_best_response(
                initial_user_state_basis=state_basis,
                opponent_util_fn=opponent_utility_fn_resp,
                opponent_value_fn=opponent_value_fn_resp,
                user_agent=deepcopy(self),
                relative_time=relative_time,
                n_uniform_sample=1,
            )
            print(f"Best response idx for IOP respond level {m}: {best_response_idx}")

            iop_trainers_respond[m].add_step(Step(
                state=state_basis,
                action=best_response_idx
            ))
            iop_trainers_respond[m].update()
            iop_trainers_respond[m].reset_step()

        self.rollout_level = None

    def _update_bayesian_mixer_respond(
        self, state_basis: np.ndarray, real_opponent_action_idx: int
    ) -> None:
        """Update Bayesian mixer untuk IOP respond."""
        iop_trainers_respond = self.trainers.get("iop_trainers_respond")
        if iop_trainers_respond is None:
            return

        iop_probs = []
        for m in range(self.M):
            iop_net = iop_trainers_respond[m].models.get("policy").get("model")
            if iop_net is None:
                iop_probs.append(1.0 / self.iop_respond_action_dim)
                continue
            iop_net.eval()
            x = torch.tensor(
                state_basis, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                logits = iop_net(x)
                probs  = torch.softmax(logits, dim=-1)
            p = float(probs[0, real_opponent_action_idx].item())
            iop_probs.append(p)

        self.bayesian_mixer_respond.update(np.array(iop_probs, dtype=np.float64))

    # ──────────────────────────────────────────────────────────────────
    # NEGMAS PROTOCOL
    # ──────────────────────────────────────────────────────────────────

    def on_negotiation_start(self, state):
        """Reset state MBOM per episode."""
        if self.rollout_level is None:
            super().on_negotiation_start(state)
            self.current_state_data_propose = None
            self.current_state_data_respond = None

    def respond(self, state, source=None) -> ResponseType:
        """
        Mekanisme respond:
        1. Flush pending propose step (next_state = state sekarang).
        2. Flush pending respond step (next_state = state sekarang).
        3. Jalankan IOP propose pipeline: step IOP propose [0] dengan offer yang diterima,
           lalu recursive imagination propose, lalu update bayesian mixer propose.
        4. Build augmented state dan sample respond action.
        5. Simpan current_state_data_respond (akan di-flush di respond() atau on_negotiation_end() berikutnya).
        """
        offer = state.current_offer

        # ── Flush pending propose step ────────────────────────────────
        if self.rollout_level is None and self.current_state_data_propose:
            next_state = object_to_dict(state)
            self.record_step_propose(
                state     = self.current_state_data_propose["state"],
                action    = self.current_state_data_propose["action"],
                utility   = self.current_state_data_propose["utility"],
                value     = self.current_state_data_propose["value"],
                log_prob  = self.current_state_data_propose["log_prob"],
                next_state = next_state,
            )
            self.current_state_data_propose = None

        # ── Flush pending respond step ────────────────────────────────
        if self.rollout_level is None and self.current_state_data_respond:
            next_state = object_to_dict(state)
            self.record_step_respond(
                state     = self.current_state_data_respond["state"],
                action    = self.current_state_data_respond["action"],
                utility   = self.current_state_data_respond["utility"],
                value     = self.current_state_data_respond["value"],
                log_prob  = self.current_state_data_respond["log_prob"],
                next_state = next_state,
            )
            self.current_state_data_respond = None

        if offer is None:
            return ResponseType.REJECT_OFFER

        t = float(getattr(state, "relative_time", 0.0))

        # ── State basis ──────────────────────────────────────────────
        state_basis = self.build_state_basis(state, offer)

        # ── MBOM: IOP propose pipeline (step IOP propose di respond()) ──
        # IOP propose di-step di respond() karena di sini kita tahu offer lawan
        if self.rollout_level is None:
            if self.opponent_state is not None:
                # offer adalah aksi riil lawan (propose dari lawan)
                opponent_propose_action_idx = self._iop_offer_to_propose_idx(offer)

                # State lawan saat dia propose = opponent_state
                state_vec = self.build_state_basis(self.opponent_state, offer)
                relative_time = float(getattr(self.opponent_state, "relative_time", 0.0))

                # Algoritma 1 baris 9: fine-tune IOP propose level-0
                iop_trainers_propose = self.trainers.get("iop_trainers_propose")
                if iop_trainers_propose is not None:
                    iop_trainers_propose[0].add_step(Step(
                        state=state_vec,
                        action=opponent_propose_action_idx
                    ))
                    iop_trainers_propose[0].update()
                    iop_trainers_propose[0].reset_step()

                    # Algoritma 1 baris 10-13: Recursive Imagination propose
                    self._run_recursive_imagination_propose(state_vec, relative_time)

                    # Algoritma 1 baris 14: update Bayesian Mixer propose
                    self._update_bayesian_mixer_propose(state_vec, opponent_propose_action_idx)

                opponent_respond_action_idx = 1  # accept/end

                state_vec = self.build_state_basis(self.opponent_state)
                relative_time = float(getattr(self.opponent_state, "relative_time", 0.0))

                iop_trainers_respond = self.trainers.get("iop_trainers_respond")
                if iop_trainers_respond is not None:
                    iop_trainers_respond[0].add_step(Step(
                        state=state_vec,
                        action=opponent_respond_action_idx
                    ))
                    iop_trainers_respond[0].update()
                    iop_trainers_respond[0].reset_step()

                    # Recursive Imagination respond
                    self._run_recursive_imagination_respond(state_vec, relative_time)

                    # Update Bayesian Mixer respond
                    self._update_bayesian_mixer_respond(state_vec, opponent_respond_action_idx)

                self.opponent_state = None

        # ── Build augmented state ────────────────────────────────────
        obs = self.build_state(state_basis)

        # ── Sample respond action ─────────────────────────────────────
        action_id, log_prob, value = self.sample_action_respond(obs)
        # action_id: 0=accept, 1=reject

        current_u = self._offer_utility(offer)

        # Simpan current respond state data
        if self.rollout_level is None:
            self.current_state_data_respond = {
                "state"   : object_to_dict(state),
                "action"  : action_id,
                "utility" : current_u,
                "value"   : value,
                "log_prob": log_prob,
            }

        if action_id == 0:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state, dest=None):
        """
        Mekanisme propose:
        1. Build augmented state dan sample propose action.
        2. Simpan current_state_data_propose (akan di-flush di respond() berikutnya).
        3. Simpan opponent_state untuk IOP propose pipeline di respond() berikutnya.
        """
        offer_context = state.current_offer

        # ── State basis ──────────────────────────────────────────────
        state_basis = self.build_state_basis(state, offer_context)

        # ── Augmented state ──────────────────────────────────────────
        obs = self.build_state(state_basis)

        # ── Sample propose action ────────────────────────────────────
        action_id, log_prob, value = self.sample_action_propose(obs)
        # action_id: 0-based index ke action_library

        offer = self.decode_action(action_id)
        current_u = self._offer_utility(offer) if offer is not None else 0.0

        # Set current_state_data_propose hanya pada mode normal
        if self.rollout_level is None:
            self.current_state_data_propose = {
                "state"   : object_to_dict(state),
                "action"  : action_id,
                "utility" : current_u,
                "value"   : value,
                "log_prob": log_prob,
            }

        # Simpan opponent_state untuk IOP propose pipeline
        self.opponent_state = state

        return offer

    # ──────────────────────────────────────────────────────────────────
    # EPISODE PAYLOAD
    # ──────────────────────────────────────────────────────────────────

    def build_episode_payload_propose(self, agreement) -> Episode:
        steps = deepcopy(self.memory_propose)
        states, actions, log_probs, values, utilities, next_states = \
            [], [], [], [], [], []

        for step in steps:
            states.append(step.state)
            actions.append(int(step.action))
            log_probs.append(0.0 if step.log_prob is None else float(step.log_prob))
            values.append(0.0 if step.value is None else float(step.value))
            utilities.append(0.0 if step.utility is None else float(step.utility))
            next_states.append(step.next_state)

        meta = {
            "agent_name" : self.name,
            "utility_min": float(self.utility_min),
            "utility_max": float(self.utility_max),
            "n_steps"    : len(steps),
            "has_agreement": agreement is not None,
            "mbom_M"           : self.M,
            "mbom_rollout_k"   : self.rollout_k,
            "bayesian_alpha_propose": self.bayesian_mixer_propose.get_alpha().tolist(),
        }

        return Episode(
            agent_name      = self.name,
            terminal_utility = self._offer_utility(agreement)
                               if agreement is not None else -1.0,
            agreement       = agreement,
            states          = states,
            actions         = actions,
            values          = values,
            log_probs       = log_probs,
            utilities       = utilities,
            q_values        = None,
            steps           = steps,
            meta            = meta,
        )

    def build_episode_payload_respond(self, agreement) -> Episode:
        steps = deepcopy(self.memory_respond)
        states, actions, log_probs, values, utilities, next_states = \
            [], [], [], [], [], []

        for step in steps:
            states.append(step.state)
            actions.append(int(step.action))
            log_probs.append(0.0 if step.log_prob is None else float(step.log_prob))
            values.append(0.0 if step.value is None else float(step.value))
            utilities.append(0.0 if step.utility is None else float(step.utility))
            next_states.append(step.next_state)

        meta = {
            "agent_name" : self.name,
            "utility_min": float(self.utility_min),
            "utility_max": float(self.utility_max),
            "n_steps"    : len(steps),
            "has_agreement": agreement is not None,
            "mbom_M"           : self.M,
            "mbom_rollout_k"   : self.rollout_k,
            "bayesian_alpha_respond": self.bayesian_mixer_respond.get_alpha().tolist(),
        }

        return Episode(
            agent_name      = self.name,
            terminal_utility = self._offer_utility(agreement)
                               if agreement is not None else -1.0,
            agreement       = agreement,
            states          = states,
            actions         = actions,
            values          = values,
            log_probs       = log_probs,
            utilities       = utilities,
            q_values        = None,
            steps           = steps,
            meta            = meta,
        )

    # Backward compat: build_episode_payload mengembalikan propose payload
    def build_episode_payload(self, agreement) -> Episode:
        return self.build_episode_payload_propose(agreement)

    # ──────────────────────────────────────────────────────────────────
    # NEGOTIATION END
    # ──────────────────────────────────────────────────────────────────

    def on_negotiation_end(self, state):
        """
        1. Jalankan IOP respond pipeline untuk ACCEPT (karena negosiasi berakhir).
        2. Flush pending propose dan respond steps.
        3. Store episodes ke propose_trainer dan respond_trainer.
        """
        if self.rollout_level is None:

            # ── IOP respond pipeline di on_negotiation_end ────────────
            # IOP respond di-step di on_negotiation_end() juga (selain di respond())
            # Aksi lawan = 0 (accept) karena negosiasi berakhir dengan deal
            # atau negosiasi berakhir karena timeout
            if self.opponent_state is not None:
                # Jika negosiasi berakhir, lawan menerima (aksi respond = 0 = accept)
                # ATAU lawan reject terakhir tapi negosiasi timeout (juga anggap action 0)
                opponent_respond_action_idx = 0  # accept/end

                state_vec = self.build_state_basis(self.opponent_state)
                relative_time = float(getattr(self.opponent_state, "relative_time", 0.0))

                iop_trainers_respond = self.trainers.get("iop_trainers_respond")
                if iop_trainers_respond is not None:
                    iop_trainers_respond[0].add_step(Step(
                        state=state_vec,
                        action=opponent_respond_action_idx
                    ))
                    iop_trainers_respond[0].update()
                    iop_trainers_respond[0].reset_step()

                    # Recursive Imagination respond
                    self._run_recursive_imagination_respond(state_vec, relative_time)

                    # Update Bayesian Mixer respond
                    self._update_bayesian_mixer_respond(state_vec, opponent_respond_action_idx)

                self.opponent_state = None

            # ── Flush pending propose step ────────────────────────────
            if self.current_state_data_propose:
                next_state = object_to_dict(state)
                self.record_step_propose(
                    state     = self.current_state_data_propose["state"],
                    action    = self.current_state_data_propose["action"],
                    utility   = self.current_state_data_propose["utility"],
                    value     = self.current_state_data_propose["value"],
                    log_prob  = self.current_state_data_propose["log_prob"],
                    next_state = next_state,
                )
                self.current_state_data_propose = None

            # ── Flush pending respond step ────────────────────────────
            if self.current_state_data_respond:
                next_state = object_to_dict(state)
                self.record_step_respond(
                    state     = self.current_state_data_respond["state"],
                    action    = self.current_state_data_respond["action"],
                    utility   = self.current_state_data_respond["utility"],
                    value     = self.current_state_data_respond["value"],
                    log_prob  = self.current_state_data_respond["log_prob"],
                    next_state = next_state,
                )
                self.current_state_data_respond = None

            agreement = getattr(state, "agreement", None)

            # ── Store episodes ke trainer masing-masing ────────────────
            if len(self.trainers) > 0:
                payload_propose = self.build_episode_payload_propose(agreement)
                payload_respond = self.build_episode_payload_respond(agreement)

                propose_trainer = self.trainers.get("propose_trainer")
                respond_trainer = self.trainers.get("respond_trainer")

                if propose_trainer is not None:
                    propose_trainer.store_episode(payload_propose)
                if respond_trainer is not None:
                    respond_trainer.store_episode(payload_respond)

    # ──────────────────────────────────────────────────────────────────
    # RESET
    # ──────────────────────────────────────────────────────────────────

    def reset_episode(self):
        self.memory_propose          = []
        self.memory_respond          = []
        self.current_state_data_propose = None
        self.current_state_data_respond = None

    # ──────────────────────────────────────────────────────────────────
    # CLONE FOR ROLLOUT
    # ──────────────────────────────────────────────────────────────────

    def clone_for_rollout(self) -> "MBOMAgent":
        """
        Buat instance MBOMAgent baru dengan konfigurasi identik tapi
        episode state bersih. Trainers di-share (bukan di-copy) agar
        clone tetap memakai model/weights yang sama dengan agent asli.
        """
        clone = MBOMAgent(
            name=self.name + "_rollout_clone",
            ufun=self.ufun,
            domain_spec=self.domain_spec,
            state_dim=self.state_dim,
            M=self.M,
            rollout_k=self.rollout_k,
            gamma=self.gamma,
            iop_batch_size=self.iop_batch_size,
            iop_update_steps=self.iop_update_steps,
            n_rollout_samples=self.n_rollout_samples,
        )

        # Share trainers
        clone.trainers = self.trainers

        # Salin rollout_level
        clone.rollout_level = self.rollout_level

        # Episode state bersih
        clone.memory_propose         = []
        clone.memory_respond         = []
        clone.current_state_data_propose = None
        clone.current_state_data_respond = None
        clone.opponent_state             = None

        return clone

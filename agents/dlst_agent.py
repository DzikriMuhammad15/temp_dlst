"""
DLSTNegotiator — agen DLST-ANESIA (Deep Learning Strategy Templates).

Mengadopsi pendekatan paper "Adaptive strategy templates using deep reinforcement
learning for multi-issue bilateral negotiation" (Bagga et al., 2025), TANPA Firefly
Algorithm untuk user modelling — karena pada project ini utility function user sudah
diketahui secara eksplisit (self.ufun).

Struktur (konsisten dengan RLNegotiator / BaseNegotiator):
  - 3 actor-critic terpisah via 3 trainer DDPG:
      * "threshold_trainer" : memprediksi ū_t (ambang batas utilitas dinamis)
      * "propose_trainer"   : bidding strategy (δ, c, p) → f_b
      * "respond_trainer"   : acceptance strategy (δ, c, p) → f_a
  - 3 memory: memory_propose, memory_respond (dari BaseNegotiator),
    dan memory_threshold_utility (khusus agen ini).
  - 1 interface build_state(state, for_=...) → build_state_propose/respond/utility_threshold.
  - sample_action_propose / sample_action_respond / sample_action_threshold_utility.

Opponent utility approximation (frequency model) MENYATU dengan reward shaper
respond. Agent mengambilnya via:
    self.trainers["respond_trainer"].reward_shaper.opponent_model
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from negmas import ResponseType

from agents.base_agent import BaseNegotiator
from data_type.step import Step
from data_type.episode import Episode
from utils.utility import compute_utility, object_to_dict

from agents.dlst.strategy_template import StrategyTemplateSpec
from agents.dlst.acceptance_strategy import AcceptanceStrategy
from agents.dlst.bidding_strategy import BiddingStrategy


class DLSTNegotiator(BaseNegotiator):

    def __init__(
        self,
        name: str,
        ufun,
        domain_spec,
        bidding_spec: StrategyTemplateSpec,
        acceptance_spec: StrategyTemplateSpec,
        u_fixed: float = 0.6,
        trainers=None,
        **kwargs,
    ):
        super().__init__(name=name, ufun=ufun, domain_spec=domain_spec, trainers=trainers, **kwargs)

        self.action_library = self._domain_attr("action_library")
        self.issue_names = self._domain_attr("issue_names")
        self.value_lists = self._domain_attr("value_lists")
        self.utility_min = float(self._domain_attr("utility_min"))
        self.utility_max = float(self._domain_attr("utility_max"))
        self.best_outcome = self._domain_attr("best_outcome")
        self.worst_outcome = self._domain_attr("worst_outcome")

        self.device = "cpu" if not torch.cuda.is_available() else "cuda"

        # ── Spec strategy template ───────────────────────────────────
        self.bidding_spec = bidding_spec
        self.acceptance_spec = acceptance_spec

        # ── Memory ketiga (khusus DLST): threshold utility ───────────
        self.memory_threshold_utility: List[Step] = []
        self.current_state_data_threshold_utility = None

        # ── Strategy template objects (OOP) ──────────────────────────
        self.acceptance_strategy = AcceptanceStrategy(acceptance_spec, u_fixed=u_fixed)
        self.bidding_strategy = BiddingStrategy(
            spec=bidding_spec,
            action_library=self.action_library,
            issue_names=self.issue_names,
            user_utility_fn=self._offer_utility,            # RAW user utility
            opponent_utility_fn=self._estimate_opponent_utility,  # estimasi U_o ∈ [0,1]
            utility_min=self.utility_min,
            utility_max=self.utility_max,
        )

        # ── Histori untuk taktik (Ω^o_t) ─────────────────────────────
        self.opponent_util_history_norm: List[float] = []  # U_u(ω^o) ternormalisasi
        self.last_received_offer = None

        # ── Cache aksi terakhir tiap arsitektur (untuk record step) ──
        self._last_threshold_value = None

    # ──────────────────────────────────────────────────────────────────
    # TRAINER ATTACH
    # ──────────────────────────────────────────────────────────────────

    def attach_trainer(self, dict_of_trainer):
        """
        agent.attach_trainer({
            "threshold_trainer": DDPGTrainer,
            "propose_trainer":   DDPGTrainer,   # bidding strategy
            "respond_trainer":   DDPGTrainer,   # acceptance strategy
        })
        """
        for key, trainer in dict_of_trainer.items():
            self.trainers[key] = trainer

    # ──────────────────────────────────────────────────────────────────
    # OPPONENT MODEL (diambil dari reward shaper respond)
    # ──────────────────────────────────────────────────────────────────

    def _opponent_model(self):
        """Ambil opponent utility approximation dari trainer respond → reward_shaper."""
        respond_trainer = self.trainers.get("respond_trainer")
        if respond_trainer is None:
            return None
        rs = getattr(respond_trainer, "reward_shaper", None)
        if rs is None:
            return None
        return getattr(rs, "opponent_model", None)

    def _estimate_opponent_utility(self, offer) -> float:
        om = self._opponent_model()
        if om is None:
            return 0.5
        return float(om.estimate_utility(offer))

    def _update_opponent_model(self, offer) -> None:
        om = self._opponent_model()
        if om is not None:
            om.update(offer)

    # ──────────────────────────────────────────────────────────────────
    # UTILITY HELPERS
    # ──────────────────────────────────────────────────────────────────

    def _offer_utility(self, offer):
        return compute_utility(self.ufun, offer, self.issue_names)

    def _normalize_utility(self, u: float) -> float:
        denom = max(1e-8, self.utility_max - self.utility_min)
        return (float(u) - self.utility_min) / denom

    # ──────────────────────────────────────────────────────────────────
    # BUILD STATE — interface tunggal dengan parameter for_
    # ──────────────────────────────────────────────────────────────────

    def build_state(self, state, offer=None, for_: str = "propose") -> np.ndarray:
        """
        Interface tunggal (konsisten dengan project). Dispatch berdasarkan for_:
          - "utility_threshold" → build_state_utility_threshold
          - "propose"           → build_state_propose (bidding)
          - "respond"           → build_state_respond (acceptance)

        Mendukung pass-through bila `state` sudah berupa vector (untuk dipakai
        trainer saat rekonstruksi state dari episode.states yang berupa dict).
        """
        if for_ == "utility_threshold":
            return self.build_state_utility_threshold(state, offer)
        elif for_ == "respond":
            return self.build_state_respond(state, offer)
        else:
            return self.build_state_propose(state, offer)

    # ── Atribut dasar (common) ───────────────────────────────────────

    def _common_attrs(self, state, offer):
        """
        Hitung atribut dasar yang dipakai semua arsitektur:
          relative_time, U_u(ω^o) norm, O_best, O_avg, O_sd, has_offer.
        Menerima state berupa dict (object_to_dict) atau SAOState.
        """
        if isinstance(state, dict):
            relative_time = float(state.get("relative_time", 0.0))
            if offer is None:
                offer = state.get("current_offer", None)
        else:
            relative_time = float(getattr(state, "relative_time", 0.0))
            if offer is None:
                offer = getattr(state, "current_offer", None)

        if offer is None:
            offer = self.best_outcome

        u_norm = self._normalize_utility(self._offer_utility(offer))
        has_offer = 1.0 if offer is not None else 0.0

        hist = self.opponent_util_history_norm
        if len(hist) > 0:
            o_best = float(max(hist))
            o_avg = float(sum(hist) / len(hist))
            o_sd = float(np.std(np.asarray(hist, dtype=np.float64)))
        else:
            o_best = u_norm
            o_avg = u_norm
            o_sd = 0.0

        return relative_time, u_norm, o_best, o_avg, o_sd, has_offer

    def build_state_utility_threshold(self, state, offer=None) -> np.ndarray:
        """
        State untuk arsitektur threshold (paper: atribut dasar).
        dim = 6: [u_norm, relative_time, O_best, O_avg, O_sd, has_offer]
        """
        if isinstance(state, (np.ndarray, list)):
            arr = np.asarray(state, dtype=np.float32)
            if arr.shape == (self.STATE_DIM_THRESHOLD,):
                return arr

        rt, u_norm, o_best, o_avg, o_sd, has_offer = self._common_attrs(state, offer)
        return np.array([u_norm, rt, o_best, o_avg, o_sd, has_offer], dtype=np.float32)

    def build_state_respond(self, state, offer=None) -> np.ndarray:
        """
        State untuk arsitektur acceptance (s^a_t):
          atribut dasar + [u_fixed, u_bar, U(ω_t) (estimasi bid agent), quantile_val]
        dim = 6 + 4 = 10.
        """
        if isinstance(state, (np.ndarray, list)):
            arr = np.asarray(state, dtype=np.float32)
            if arr.shape == (self.STATE_DIM_RESPOND,):
                return arr

        rt, u_norm, o_best, o_avg, o_sd, has_offer = self._common_attrs(state, offer)

        u_fixed = float(self.acceptance_strategy.u_fixed)
        u_bar = float(self._predict_threshold(state, offer))
        # Estimasi utilitas bid yang akan diusulkan agent (pakai best bid sbg proxy ringan)
        u_future = float(self._u_future_bid_norm())
        # Kuantil sederhana (median) dari histori utilitas lawan
        hist = self.opponent_util_history_norm
        quantile_val = float(np.quantile(np.asarray(hist), 0.5)) if len(hist) > 0 else u_norm

        base = [u_norm, rt, o_best, o_avg, o_sd, has_offer]
        extra = [u_fixed, u_bar, u_future, quantile_val]
        return np.array(base + extra, dtype=np.float32)

    def build_state_propose(self, state, offer=None) -> np.ndarray:
        """
        State untuk arsitektur bidding (s^b_t):
          atribut dasar + [u_bar, U_boulware, U_pareto, U_bopp]
          (utilitas-utilitas dari taktik bidding sebagai 'saran', ternormalisasi)
        dim = 6 + 4 = 10.
        """
        if isinstance(state, (np.ndarray, list)):
            arr = np.asarray(state, dtype=np.float32)
            if arr.shape == (self.STATE_DIM_PROPOSE,):
                return arr

        rt, u_norm, o_best, o_avg, o_sd, has_offer = self._common_attrs(state, offer)

        u_bar = float(self._predict_threshold(state, offer))

        # "Saran" taktik (utilitas user ternormalisasi dari bid tiap taktik standar).
        ctx = {
            "u_bar": u_bar,
            "opponent_last_offer": self.last_received_offer,
        }
        # Pakai parameter taktik default ringan (a,b) untuk fitur state.
        try:
            b_boul = self.bidding_strategy._tactic_boulware(rt, [0.1], ctx)
            b_par = self.bidding_strategy._tactic_pareto(rt, [-0.5, 0.8], ctx)
            b_opp = self.bidding_strategy._tactic_b_opp(rt, [], ctx)
            u_boul = self._normalize_utility(self._offer_utility(b_boul))
            u_par = self._normalize_utility(self._offer_utility(b_par))
            u_bopp = self._normalize_utility(self._offer_utility(b_opp))
        except Exception:
            u_boul = u_par = u_bopp = u_norm

        base = [u_norm, rt, o_best, o_avg, o_sd, has_offer]
        extra = [u_bar, u_boul, u_par, u_bopp]
        return np.array(base + extra, dtype=np.float32)

    # State dims (dipakai juga oleh main script untuk membangun network)
    STATE_DIM_THRESHOLD = 6
    STATE_DIM_RESPOND = 10
    STATE_DIM_PROPOSE = 10

    def _u_future_bid_norm(self) -> float:
        """Utilitas (norm) bid terbaik agent sebagai proxy U(ω_t)."""
        best = max(self.action_library, key=lambda o: self._offer_utility(o)) if self.action_library else None
        return self._normalize_utility(self._offer_utility(best)) if best is not None else 0.0

    # ──────────────────────────────────────────────────────────────────
    # THRESHOLD PREDICTION (actor threshold)
    # ──────────────────────────────────────────────────────────────────

    def _predict_threshold(self, state, offer) -> float:
        """
        Forward actor threshold → ū_t ∈ [0,1]. Dipakai sebagai input bagi
        state acceptance/bidding dan sebagai taktik u_bar. Bila trainer threshold
        belum ada, fallback ke 0.5. Untuk menghindari rekursi tak hingga
        (build_state_threshold tidak memanggil _predict_threshold), aman.
        """
        threshold_trainer = self.trainers.get("threshold_trainer")
        if threshold_trainer is None:
            return 0.5
        actor_entry = threshold_trainer.models.get("actor")
        if actor_entry is None:
            return 0.5
        actor = actor_entry["model"]

        obs = self.build_state_utility_threshold(state, offer)
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actor.eval()
        with torch.no_grad():
            u_bar = actor(x)
        return float(u_bar.view(-1)[0].item())

    # ──────────────────────────────────────────────────────────────────
    # SAMPLE ACTIONS — 3 arsitektur
    # ──────────────────────────────────────────────────────────────────

    def sample_action_threshold_utility(self, obs):
        """
        Sample aksi threshold (deterministic DDPG): output ū_t ∈ [0,1].
        Return (action_vector(list[float]), value(None)). value None karena
        DDPG critic butuh (s,a); value tidak diperlukan saat acting.
        """
        threshold_trainer = self.trainers.get("threshold_trainer")
        if threshold_trainer is None:
            raise ValueError("No threshold_trainer attached. Attach via attach_trainer.")
        actor = threshold_trainer.models.get("actor").get("model")
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actor.eval()
        with torch.no_grad():
            a = actor(x)
        return a.view(-1).cpu().numpy().tolist()

    def sample_action_propose(self, obs):
        """
        Sample aksi bidding strategy (DDPG): output vektor (δ, c, p) flat.
        Return list[float] panjang bidding_spec.action_dim.
        """
        propose_trainer = self.trainers.get("propose_trainer")
        if propose_trainer is None:
            raise ValueError("No propose_trainer attached. Attach via attach_trainer.")
        actor = propose_trainer.models.get("actor").get("model")
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actor.eval()
        with torch.no_grad():
            a = actor(x)
        return a.view(-1).cpu().numpy().tolist()

    def sample_action_respond(self, obs):
        """
        Sample aksi acceptance strategy (DDPG): output vektor (δ, c, p) flat.
        Return list[float] panjang acceptance_spec.action_dim.
        """
        respond_trainer = self.trainers.get("respond_trainer")
        if respond_trainer is None:
            raise ValueError("No respond_trainer attached. Attach via attach_trainer.")
        actor = respond_trainer.models.get("actor").get("model")
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actor.eval()
        with torch.no_grad():
            a = actor(x)
        return a.view(-1).cpu().numpy().tolist()

    # ──────────────────────────────────────────────────────────────────
    # RECORD STEP — memory ketiga (threshold utility)
    # ──────────────────────────────────────────────────────────────────

    def record_step_threshold_utility(
        self, state=None, action=None, utility=None,
        value=None, q_value=None, log_prob=None, next_state=None
    ):
        self.memory_threshold_utility.append(Step(
            state=state, action=action, utility=utility, value=value,
            q_value=q_value, log_prob=log_prob, next_state=next_state,
        ))

    # ──────────────────────────────────────────────────────────────────
    # NEGMAS PROTOCOL
    # ──────────────────────────────────────────────────────────────────

    def on_negotiation_start(self, state):
        super().on_negotiation_start(state)
        self.memory_threshold_utility = []
        self.current_state_data_propose = None
        self.current_state_data_respond = None
        self.current_state_data_threshold_utility = None
        self.opponent_util_history_norm = []
        self.last_received_offer = None

    def _flush_propose(self, state):
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

    def _flush_respond(self, state):
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

    def _flush_threshold(self, state, utility_override=None):
        """
        Flush threshold step. Bila utility_override diberikan (di
        on_negotiation_end), gunakan itu sebagai utility (utilitas agreement
        atau -1). Bila None (flush di respond), pakai utility tersimpan
        (= utilitas offer lawan), sama seperti respond.
        """
        if self.current_state_data_threshold_utility:
            next_state = object_to_dict(state)
            util = (
                utility_override
                if utility_override is not None
                else self.current_state_data_threshold_utility["utility"]
            )
            self.record_step_threshold_utility(
                state=self.current_state_data_threshold_utility["state"],
                action=self.current_state_data_threshold_utility["action"],
                utility=util,
                value=self.current_state_data_threshold_utility["value"],
                log_prob=self.current_state_data_threshold_utility["log_prob"],
                next_state=next_state,
            )
            self.current_state_data_threshold_utility = None

    def respond(self, state, source=None) -> ResponseType:
        offer = state.current_offer

        # ── 1. Flush pending steps (awal respond) ────────────────────
        self._flush_propose(state)
        self._flush_respond(state)
        self._flush_threshold(state)  # utility = utilitas offer lawan (tersimpan)

        if offer is None:
            return ResponseType.REJECT_OFFER

        # ── 2. Update opponent model & histori ───────────────────────
        self._update_opponent_model(offer)
        u_user_norm = self._normalize_utility(self._offer_utility(offer))
        self.opponent_util_history_norm.append(u_user_norm)
        self.last_received_offer = offer

        t = float(getattr(state, "relative_time", 0.0))

        # ── 3. Prediksi ū_t (threshold actor) ────────────────────────
        u_bar = self._predict_threshold(state, offer)
        self._last_threshold_value = u_bar

        # ── 4. Sample aksi acceptance (respond) ──────────────────────
        obs_respond = self.build_state_respond(state, offer)
        action_respond = self.sample_action_respond(obs_respond)

        # ── 5. Decide accept/reject via acceptance template (f_a) ────
        ctx_acc = {
            "u_bar": u_bar,
            "u_future_bid": self._u_future_bid_norm(),
            "opponent_util_history": list(self.opponent_util_history_norm),
        }
        accept = self.acceptance_strategy.decide(t, action_respond, u_user_norm, ctx_acc)
        action_id_respond = 0 if accept else 1  # 0=accept, 1=reject (untuk reward shaper)

        # ── 6. Simpan current respond step data ──────────────────────
        # utility yang disimpan = utilitas offer lawan (U_u(ω^o)) — RAW.
        current_u = self._offer_utility(offer)
        self.current_state_data_respond = {
            "state": object_to_dict(state),
            "action": action_respond,   # vektor continuous (untuk DDPG)
            "utility": current_u,
            "value": None,
            "log_prob": None,
            # Simpan juga keputusan diskrit untuk konsistensi reward (accept/reject).
            "decision": action_id_respond,
        }

        # ── 7. Simpan current threshold step data ────────────────────
        # Diambil dari respond() (sama seperti respond). Action = ū_t (vektor [ū_t]).
        # utility (saat flush di respond berikutnya) = utilitas offer lawan.
        obs_threshold = self.build_state_utility_threshold(state, offer)
        action_threshold = self.sample_action_threshold_utility(obs_threshold)
        self.current_state_data_threshold_utility = {
            "state": object_to_dict(state),
            "action": action_threshold,  # vektor [ū_t]
            "utility": current_u,         # utilitas offer lawan (untuk flush di respond)
            "value": None,
            "log_prob": None,
        }

        if accept:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state, dest=None):
        offer_opponent = state.current_offer
        t = float(getattr(state, "relative_time", 0.0))

        # ── Prediksi ū_t & sample aksi bidding ───────────────────────
        u_bar = self._predict_threshold(state, offer_opponent)
        obs_propose = self.build_state_propose(state, offer_opponent)
        action_propose = self.sample_action_propose(obs_propose)

        # ── Hasilkan offer via bidding template (f_b) ────────────────
        ctx_bid = {
            "u_bar": u_bar,
            "opponent_last_offer": self.last_received_offer,
        }
        offer = self.bidding_strategy.generate(t, action_propose, ctx_bid)
        current_u = self._offer_utility(offer)

        # ── Simpan current propose step data ─────────────────────────
        self.current_state_data_propose = {
            "state": object_to_dict(state),
            "action": action_propose,  # vektor continuous (untuk DDPG)
            "utility": current_u,
            "value": None,
            "log_prob": None,
        }
        return offer

    def on_negotiation_end(self, state):
        agreement = getattr(state, "agreement", None)

        # ── Flush pending propose & respond (utility tersimpan) ──────
        self._flush_propose(state)
        self._flush_respond(state)

        # ── Flush threshold dengan utility khusus ────────────────────
        # Saat on_negotiation_end: utility = utilitas agreement bila ada,
        # atau -1 bila tidak ada agreement (sesuai spesifikasi).
        if agreement is not None:
            util_threshold = self._offer_utility(agreement)
        else:
            util_threshold = -1.0
        self._flush_threshold(state, utility_override=util_threshold)

        # ── Build payloads & store ke trainer masing-masing ──────────
        if len(self.trainers) > 0:
            payload_propose = self.build_episode_payload_propose(agreement)
            payload_respond = self.build_episode_payload_respond(agreement)
            payload_threshold = self.build_episode_payload_threshold(agreement)

            propose_trainer = self.trainers.get("propose_trainer")
            respond_trainer = self.trainers.get("respond_trainer")
            threshold_trainer = self.trainers.get("threshold_trainer")

            if propose_trainer is not None:
                propose_trainer.store_episode(payload_propose)
            if respond_trainer is not None:
                respond_trainer.store_episode(payload_respond)
            if threshold_trainer is not None:
                threshold_trainer.store_episode(payload_threshold)

    # ──────────────────────────────────────────────────────────────────
    # EPISODE PAYLOADS — 3 jenis (propose, respond, threshold)
    # ──────────────────────────────────────────────────────────────────

    def _build_payload_from_memory(self, memory, agreement):
        steps = deepcopy(memory)
        states, actions, log_probs, values, utilities = [], [], [], [], []

        for step in steps:
            states.append(step.state)
            # action berupa vektor continuous (DDPG) — simpan apa adanya
            actions.append(step.action)
            log_probs.append(0.0 if step.log_prob is None else float(step.log_prob))
            values.append(0.0 if step.value is None else float(step.value))
            utilities.append(0.0 if step.utility is None else float(step.utility))

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

    def build_episode_payload_propose(self, agreement):
        return self._build_payload_from_memory(self.memory_propose, agreement)

    def build_episode_payload_respond(self, agreement):
        return self._build_payload_from_memory(self.memory_respond, agreement)

    def build_episode_payload_threshold(self, agreement):
        return self._build_payload_from_memory(self.memory_threshold_utility, agreement)

    # Backward compat
    def build_episode_payload(self, agreement):
        return self.build_episode_payload_propose(agreement)

    # ──────────────────────────────────────────────────────────────────
    # ABSTRACT METHODS (interface NegosiatorFunctionality)
    # ──────────────────────────────────────────────────────────────────

    def decode_action(self, action_id: int):
        """
        Untuk DLST, offer dihasilkan oleh bidding template, bukan index langsung.
        Disediakan untuk memenuhi interface; mengembalikan bid pada index aman.
        """
        idx = min(int(action_id), len(self.action_library) - 1) if self.action_library else 0
        return self.action_library[idx] if self.action_library else None

    def _offer_to_action_idx(self, offer):
        if offer is None:
            return 0
        try:
            return self.action_library.index(offer)
        except ValueError:
            return 0

    def _mask_logits(self, logits, allow_accept: bool):
        # DDPG tidak memakai logits diskrit; disediakan untuk memenuhi interface.
        return logits

    def reset_episode(self):
        self.memory_propose = []
        self.memory_respond = []
        self.memory_threshold_utility = []
        self.current_state_data_propose = None
        self.current_state_data_respond = None
        self.current_state_data_threshold_utility = None
        self.opponent_util_history_norm = []
        self.last_received_offer = None

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


# =============================================================================
# RESPOND ACTION CONSTANTS
# Sesuai dengan encoding di seluruh agent (mbom_agent.py, rl_agent.py, dll):
#   action 0 = ACCEPT_OFFER
#   action 1 = REJECT_OFFER  (lanjut negosiasi / counter-offer)
# =============================================================================
_RESPOND_ACTION_ACCEPT = 0
_RESPOND_ACTION_REJECT = 1


# =============================================================================
# ProposeRewardConfig
# Konfigurasi reward khusus untuk propose trainer.
# =============================================================================
@dataclass
class ProposeRewardConfig:
    """
    Konfigurasi reward untuk propose trainer.

    Mode yang tersedia:
      - mode1: r_bid (Persamaan 9)
                 Reward = U_u(omega_acc, t)  jika terjadi kesepakatan pada step ini
                          -1                  selain itu (negosiasi masih berjalan)
    """
    mode: str = "mode1"         # "mode1" | (tambahkan mode baru di masa depan)
    utility_scale: bool = True  # apakah utility dinormalisasi ke [0, 1]


# =============================================================================
# RespondRewardConfig
# Konfigurasi reward khusus untuk respond trainer.
# =============================================================================
@dataclass
class RespondRewardConfig:
    """
    Konfigurasi reward untuk respond trainer.

    Setiap step respond memiliki tiga kemungkinan "event":
      1. Accept    : agen menerima tawaran lawan  (action == 0)
      2. Offer     : agen menolak dan melanjutkan negosiasi (action == 1, bukan step terakhir)
      3. Failure   : negosiasi berakhir tanpa kesepakatan (step terakhir, agreement=None)

    Reward masing-masing event ditentukan oleh kombinasi formula berikut.

    ──────────────────────────────────────────────────────────────────
    accept_variant   → formula r_accept_*:
      "a" : r_accept_a = U_A(omega^{t-1}_{B→A})
            Utility langsung dari tawaran yang diterima.
      "b" : r_accept_b = (tanh(5*(U_A(omega^{t-1}_{B→A}) - 0.5)) + 1) / 2
            Hyperbolic-tangent: < 0.5 → reward rendah, > 0.5 → reward tinggi.
      "c" : r_accept_c = tanh(5*(U_A(omega^{t-1}_{B→A}) - 0.5))
            Perluasan (b) ke arah negatif: utility=0 → ~-1, utility=0.5 → 0.

    offer_variant    → formula r_offer_*:
      "a" : r_offer_a = 0
            Tidak ada reward saat negosiasi berlanjut.
      "b" : r_offer_b = (1 - U_A(omega^{t-1}_{B→A})) / 100
            Mendorong agen tidak terburu-buru menerima tawaran rendah.
      "c" : r_offer_c = (1 - average_10) / 100
            Menggunakan rata-rata 10 utilitas tawaran terakhir dari lawan.

    failure_penalty  → penalti saat negosiasi gagal:
      -1   | -0.5 | 0

    ──────────────────────────────────────────────────────────────────
    Daftar mode (accept_variant, offer_variant, failure_penalty):
    ──────────────────────────────────────────────────────────────────
      mode1  : (a, a, -1)     mode10 : (a, a, -0.5)   mode19 : (a, a,  0)
      mode2  : (b, a, -1)     mode11 : (b, a, -0.5)   mode20 : (b, a,  0)
      mode3  : (c, a, -1)     mode12 : (c, a, -0.5)   mode21 : (c, a,  0)
      mode4  : (a, b, -1)     mode13 : (a, b, -0.5)   mode22 : (a, b,  0)
      mode5  : (b, b, -1)     mode14 : (b, b, -0.5)   mode23 : (b, b,  0)
      mode6  : (c, b, -1)     mode15 : (c, b, -0.5)   mode24 : (c, b,  0)
      mode7  : (a, c, -1)     mode16 : (a, c, -0.5)   mode25 : (a, c,  0)
      mode8  : (b, c, -1)     mode17 : (b, c, -0.5)   mode26 : (b, c,  0)
      mode9  : (c, c, -1)     mode18 : (c, c, -0.5)   mode27 : (c, c,  0)

    # ── TAMBAHKAN MODE RESPOND BARU DI SINI ─────────────────────────
    # 1. Tambahkan entri di _RESPOND_MODE_TABLE di dalam class RewardShaper:
    #    "modeN": (accept_variant, offer_variant, failure_penalty)
    #    Contoh: "mode28": ("a", "a", -2.0)
    # 2. Jika perlu formula accept/offer BARU (selain a/b/c), tambahkan:
    #    - _r_accept_<huruf>(self, u_accept) -> float
    #    - _r_offer_<huruf>(self, u_current, recent_utilities) -> float
    #    lalu daftarkan di _RESPOND_ACCEPT_FNS / _RESPOND_OFFER_FNS.
    # ────────────────────────────────────────────────────────────────
    """
    mode: str = "mode1"
    utility_scale: bool = True
    offer_avg_window: int = 10  # ukuran window rata-rata untuk offer_variant "c"


# =============================================================================
# ThresholdRewardConfig  (DLST-ANESIA)
# Konfigurasi reward khusus untuk utility-threshold trainer.
# =============================================================================
@dataclass
class ThresholdRewardConfig:
    """
    Konfigurasi reward untuk utility-threshold trainer (DLST-ANESIA, Persamaan 8).

    Reward ambang batas utilitas (r_ū_t):
      r_ū_t = U_u(ω_acc, t)   jika terjadi kesepakatan
              U_u(ω^o_t, t)    jika menerima tawaran lawan
              -1               selain itu

    Mode yang tersedia:
      - mode1: rumus Persamaan 8 di atas (default DLST-ANESIA).

    # ── TAMBAHKAN MODE THRESHOLD BARU DI SINI ───────────────────────
    # Tambahkan cabang baru di _compute_step_reward_threshold() pada RewardShaper.
    """
    mode: str = "mode1"
    utility_scale: bool = True


# =============================================================================
# RewardShaper
# =============================================================================
class RewardShaper:
    """
    Kelas tunggal untuk menghitung reward per-step dari data episode.

    Instansiasi:
        RewardShaper(cfg, utility_min, utility_max, for_="propose")
        RewardShaper(cfg, utility_min, utility_max, for_="respond", opponent_model=...)
        RewardShaper(cfg, utility_min, utility_max, for_="utility_threshold")

    Parameter `for_` WAJIB diisi dan menentukan jenis reward yang digunakan:
      - "propose"           : menggunakan ProposeRewardConfig
      - "respond"           : menggunakan RespondRewardConfig
      - "utility_threshold" : menggunakan ThresholdRewardConfig (DLST-ANESIA)

    Opponent utility approximation (DLST-ANESIA):
      Reward respond (r_acc, Persamaan 10) bergantung pada utilitas lawan U_o.
      Karena U_o tidak diketahui agent, sebuah FrequencyOpponentModel MENYATU
      dengan reward shaper respond melalui parameter `opponent_model`. Object ini
      dapat diakses agent via: trainer_respond.reward_shaper.opponent_model
      (mis. untuk parameter template strategy / Pareto bidding).
    """

    # ── Tabel mode untuk respond ─────────────────────────────────────
    # Format: mode_name -> (accept_variant, offer_variant, failure_penalty)
    #
    # ── TAMBAHKAN MODE RESPOND BARU DI SINI ─────────────────────────
    _RESPOND_MODE_TABLE: Dict[str, tuple] = {
        "mode1":  ("a", "a", -1.0),
        "mode2":  ("b", "a", -1.0),
        "mode3":  ("c", "a", -1.0),
        "mode4":  ("a", "b", -1.0),
        "mode5":  ("b", "b", -1.0),
        "mode6":  ("c", "b", -1.0),
        "mode7":  ("a", "c", -1.0),
        "mode8":  ("b", "c", -1.0),
        "mode9":  ("c", "c", -1.0),
        "mode10": ("a", "a", -0.5),
        "mode11": ("b", "a", -0.5),
        "mode12": ("c", "a", -0.5),
        "mode13": ("a", "b", -0.5),
        "mode14": ("b", "b", -0.5),
        "mode15": ("c", "b", -0.5),
        "mode16": ("a", "c", -0.5),
        "mode17": ("b", "c", -0.5),
        "mode18": ("c", "c", -0.5),
        "mode19": ("a", "a",  0.0),
        "mode20": ("b", "a",  0.0),
        "mode21": ("c", "a",  0.0),
        "mode22": ("a", "b",  0.0),
        "mode23": ("b", "b",  0.0),
        "mode24": ("c", "b",  0.0),
        "mode25": ("a", "c",  0.0),
        "mode26": ("b", "c",  0.0),
        "mode27": ("c", "c",  0.0),
        # ── Tambahkan mode baru di bawah baris ini ──────────────────
        # "mode28": ("a", "a", -2.0),
    }

    # ── Mode respond khusus DLST-ANESIA (Persamaan 10) ───────────────
    # Mode ini TIDAK memakai kombinasi (accept_var, offer_var, penalty) di atas,
    # melainkan rumus r_acc dari paper yang membutuhkan U_o (opponent utility).
    # Diproses terpisah di _compute_step_reward_respond().
    #   "dlst": r_acc (Persamaan 10)
    _RESPOND_DLST_MODES = ("dlst",)

    def __init__(
        self,
        cfg,
        utility_min: float = 0.0,
        utility_max: float = 1.0,
        for_: Literal["propose", "respond", "utility_threshold"] = None,
        opponent_model: Optional[Any] = None,
    ):
        """
        Parameters
        ----------
        cfg : ProposeRewardConfig | RespondRewardConfig | ThresholdRewardConfig
            Konfigurasi reward.
        utility_min : float
            Nilai utilitas minimum domain (untuk normalisasi).
        utility_max : float
            Nilai utilitas maksimum domain (untuk normalisasi).
        for_ : "propose" | "respond" | "utility_threshold"
            Wajib diisi. Menentukan jenis reward shaper.
        opponent_model : FrequencyOpponentModel | None
            Hanya relevan untuk for_="respond". Digunakan untuk mengestimasi
            U_o pada reward r_acc (Persamaan 10) dan dapat diambil agent untuk
            keperluan strategi (mis. Pareto bidding). Jika None pada respond,
            reward r_acc akan fallback ke variant accept/offer murni
            (tanpa pembanding U_o).
        """
        if for_ not in ("propose", "respond", "utility_threshold"):
            raise ValueError(
                f"[RewardShaper] Parameter 'for_' wajib diisi dengan "
                f"'propose', 'respond', atau 'utility_threshold', bukan: {for_!r}"
            )
        if for_ == "propose" and not isinstance(cfg, ProposeRewardConfig):
            raise TypeError(
                "[RewardShaper] for_='propose' membutuhkan cfg bertipe ProposeRewardConfig."
            )
        if for_ == "respond" and not isinstance(cfg, RespondRewardConfig):
            raise TypeError(
                "[RewardShaper] for_='respond' membutuhkan cfg bertipe RespondRewardConfig."
            )
        if for_ == "utility_threshold" and not isinstance(cfg, ThresholdRewardConfig):
            raise TypeError(
                "[RewardShaper] for_='utility_threshold' membutuhkan cfg bertipe ThresholdRewardConfig."
            )

        self.cfg = cfg
        self.utility_min = utility_min
        self.utility_max = utility_max
        self.for_ = for_
        # Opponent utility approximation menyatu dengan respond shaper (DLST-ANESIA).
        self.opponent_model = opponent_model

    # ─────────────────────────────────────────────────────────────────
    # UTILITY SCALING
    # ─────────────────────────────────────────────────────────────────

    def _scale(self, u: float) -> float:
        """Normalisasi utilitas ke [0, 1] berdasarkan utility_min dan utility_max."""
        if not self.cfg.utility_scale:
            return float(u)
        denom = max(1e-8, self.utility_max - self.utility_min)
        return (float(u) - self.utility_min) / denom

    # ─────────────────────────────────────────────────────────────────
    # PROPOSE — FORMULA
    # ─────────────────────────────────────────────────────────────────

    def _compute_propose_mode1(self, u: float, is_agreement_step: bool) -> float:
        """
        Persamaan 9 (r_bid):
          r_bid = U_u(omega_acc, t)  jika terjadi kesepakatan pada step ini
                  -1                  selain itu
        """
        if is_agreement_step:
            return u        # U_u(omega_acc, t) — sudah di-scale
        return -1.0

    # ─────────────────────────────────────────────────────────────────
    # RESPOND — FORMULA ACCEPT (r_accept_*)
    # ─────────────────────────────────────────────────────────────────

    def _r_accept_a(self, u_accept: float) -> float:
        """
        Rumus (4-a): r_accept_a = U_A(omega^{t-1}_{B→A})
        Utility langsung dari tawaran yang diterima.
        """
        return u_accept

    def _r_accept_b(self, u_accept: float) -> float:
        """
        Rumus (4-b): r_accept_b = (tanh(5*(U_A - 0.5)) + 1) / 2
        Reward < 0.5 jika utility < 0.5, reward > 0.5 jika utility > 0.5.
        """
        return (math.tanh(5.0 * (u_accept - 0.5)) + 1.0) / 2.0

    def _r_accept_c(self, u_accept: float) -> float:
        """
        Rumus (4-c): r_accept_c = tanh(5*(U_A - 0.5))
        Perluasan ke arah negatif: utility=0 → ~-1, utility=0.5 → 0.
        """
        return math.tanh(5.0 * (u_accept - 0.5))

    # ─────────────────────────────────────────────────────────────────
    # RESPOND — FORMULA OFFER / CONTINUED NEGOTIATION (r_offer_*)
    # ─────────────────────────────────────────────────────────────────

    def _r_offer_a(self, u_current: float, recent_utilities: List[float]) -> float:
        """
        Rumus (5-a): r_offer_a = 0
        Tidak ada reward saat negosiasi berlanjut.
        """
        return 0.0

    def _r_offer_b(self, u_current: float, recent_utilities: List[float]) -> float:
        """
        Rumus (5-b): r_offer_b = (1 - U_A(omega^{t-1}_{B→A})) / 100
        Mendorong agen tidak terburu-buru menerima tawaran rendah dari lawan.
        Skala /100 agar akumulasi tidak mengganggu reward terminal.
        """
        return (1.0 - u_current) / 100.0

    def _r_offer_c(self, u_current: float, recent_utilities: List[float]) -> float:
        """
        Rumus (5-c): r_offer_c = (1 - average_10) / 100
        Menggunakan rata-rata utilitas dari N tawaran terakhir lawan.
        Skala /100 agar akumulasi tidak mengganggu reward terminal.
        """
        if len(recent_utilities) == 0:
            avg = 0.0
        else:
            avg = sum(recent_utilities) / len(recent_utilities)
        return (1.0 - avg) / 100.0

    # ─────────────────────────────────────────────────────────────────
    # DISPATCH TABLE UNTUK RESPOND FORMULAS
    # ─────────────────────────────────────────────────────────────────

    @property
    def _RESPOND_ACCEPT_FNS(self) -> Dict[str, Any]:
        # ── Tambahkan formula accept baru di sini ───────────────────
        return {
            "a": self._r_accept_a,
            "b": self._r_accept_b,
            "c": self._r_accept_c,
        }

    @property
    def _RESPOND_OFFER_FNS(self) -> Dict[str, Any]:
        # ── Tambahkan formula offer baru di sini ────────────────────
        return {
            "a": self._r_offer_a,
            "b": self._r_offer_b,
            "c": self._r_offer_c,
        }

    # ─────────────────────────────────────────────────────────────────
    # PROPOSE — COMPUTE STEP REWARD
    # ─────────────────────────────────────────────────────────────────

    def _compute_step_reward_propose(
        self,
        step: Any,
        episode: Any,
        step_idx: int,
    ) -> float:
        """
        Hitung reward untuk satu step pada trainer propose.
        Menggunakan ProposeRewardConfig.mode.
        """
        u = step.utility
        if u is None:
            u = 0.0
        u = self._scale(float(u))

        is_last = step_idx == len(episode.steps) - 1
        is_agreement_step = is_last and (episode.agreement is not None)

        mode = self.cfg.mode.lower()

        if mode == "mode1":
            return self._compute_propose_mode1(u, is_agreement_step)

        # ── TAMBAHKAN MODE PROPOSE BARU DI SINI ─────────────────────
        # elif mode == "mode2":
        #     return self._compute_propose_mode2(u, step_idx, is_agreement_step)
        # ────────────────────────────────────────────────────────────

        raise ValueError(
            f"[RewardShaper] mode propose tidak dikenal: '{mode}'. "
            f"Mode yang tersedia: mode1"
        )

    # ─────────────────────────────────────────────────────────────────
    # RESPOND — COMPUTE STEP REWARD
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _action_is_accept(action: Any) -> Optional[bool]:
        """
        Tentukan apakah `action` menyatakan ACCEPT (0) atau REJECT (1) untuk
        respond diskrit. Untuk DLST, action berupa vektor continuous → tidak
        bisa di-decode menjadi accept/reject langsung; kembalikan None agar
        caller memakai heuristik (is_last & has_agreement).
        """
        if action is None:
            return None
        if isinstance(action, (int, float)) and not isinstance(action, bool):
            ai = int(action)
            if ai == _RESPOND_ACTION_ACCEPT:
                return True
            if ai == _RESPOND_ACTION_REJECT:
                return False
            return None
        # list/tuple/ndarray (DLST continuous action) → tak terdefinisi
        return None

    def _compute_step_reward_respond(
        self,
        step: Any,
        episode: Any,
        step_idx: int,
    ) -> float:
        """
        Hitung reward untuk satu step pada trainer respond.

        Dua jalur:
          - mode "dlst"        : r_acc (Persamaan 10), butuh U_o dari opponent_model.
          - mode "modeN"       : kombinasi (accept_variant, offer_variant, failure_penalty).
        """
        mode = self.cfg.mode.lower()

        # ── Jalur DLST-ANESIA (Persamaan 10) ─────────────────────────
        if mode in self._RESPOND_DLST_MODES:
            return self._compute_respond_dlst(step, episode, step_idx)

        # ── Jalur kombinasi (mode1..mode27) ──────────────────────────
        window = self.cfg.offer_avg_window

        if mode not in self._RESPOND_MODE_TABLE:
            raise ValueError(
                f"[RewardShaper] mode respond tidak dikenal: '{mode}'. "
                f"Mode yang tersedia: {list(self._RESPOND_MODE_TABLE.keys())} "
                f"+ {list(self._RESPOND_DLST_MODES)}"
            )

        accept_var, offer_var, failure_penalty = self._RESPOND_MODE_TABLE[mode]
        accept_fn = self._RESPOND_ACCEPT_FNS[accept_var]
        offer_fn  = self._RESPOND_OFFER_FNS[offer_var]

        u = step.utility
        if u is None:
            u = 0.0
        u = self._scale(float(u))

        action = step.action
        is_last = step_idx == len(episode.steps) - 1
        has_agreement = episode.agreement is not None

        # ── Kasus 3: Gagal (negosiasi berakhir tanpa agreement) ──────
        if is_last and not has_agreement:
            return float(failure_penalty)

        # ── Kasus 1: Accept ──────────────────────────────────────────
        # Accept terjadi ketika: action menyatakan ACCEPT, atau (step terakhir
        # & ada agreement). Untuk action vektor (decode None), pakai heuristik.
        decoded = self._action_is_accept(action)
        is_accept = (decoded is True) or (is_last and has_agreement)
        if is_accept:
            return accept_fn(u)

        # ── Kasus 2: Offer / Continued Negotiation ───────────────────
        recent_utilities: List[float] = []
        start = max(0, step_idx - window)
        for prev_idx in range(start, step_idx):
            prev_u = episode.steps[prev_idx].utility
            if prev_u is None:
                prev_u = 0.0
            recent_utilities.append(self._scale(float(prev_u)))

        return offer_fn(u, recent_utilities)

    # ─────────────────────────────────────────────────────────────────
    # RESPOND — DLST-ANESIA (Persamaan 10, r_acc)
    # ─────────────────────────────────────────────────────────────────

    def _offer_from_step(self, step: Any) -> Any:
        """Ambil offer lawan dari state dict step (current_offer)."""
        st = step.state
        if isinstance(st, dict):
            return st.get("current_offer", None)
        return getattr(st, "current_offer", None)

    def _estimate_opponent_utility(self, step: Any) -> Optional[float]:
        """Estimasi U_o(ω^o_t) via opponent_model bila tersedia."""
        if self.opponent_model is None:
            return None
        offer = self._offer_from_step(step)
        return float(self.opponent_model.estimate_utility(offer))

    def _compute_respond_dlst(
        self,
        step: Any,
        episode: Any,
        step_idx: int,
    ) -> float:
        """
        Persamaan 10 (r_acc):
          r_acc = U_u(ω_acc, t)   jika sepakat   & U_o(ω_acc, t) <= U_u(ω_acc, t)
                  U_u(ω^o_t, t)    jika menolak  & U_o(ω^o_t, t) >= U_u(ω^o_t, t)
                  -1               selain itu

        U_u (user utility dari offer lawan) = step.utility (sudah di-scale).
        U_o (opponent utility dari offer lawan) = opponent_model.estimate_utility.
        Bila opponent_model None, U_o dianggap tidak melanggar syarat (fallback
        ke pemberian reward positif sesuai utilitas, tanpa pembanding).
        """
        u_user = step.utility
        if u_user is None:
            u_user = 0.0
        u_user = self._scale(float(u_user))

        u_opp = self._estimate_opponent_utility(step)  # sudah di [0,1] atau None

        action = step.action
        is_last = step_idx == len(episode.steps) - 1
        has_agreement = episode.agreement is not None

        decoded = self._action_is_accept(action)
        # Accept: action ACCEPT, atau (step terakhir & ada agreement).
        is_accept = (decoded is True) or (is_last and has_agreement)
        # Reject: action REJECT (atau action vektor di step non-terminal) &
        # bukan accept terminal.
        is_reject = (not is_accept) and not (is_last and not has_agreement)

        # ── Kasus sepakat & U_o <= U_u ───────────────────────────────
        if is_accept:
            if u_opp is None or u_opp <= u_user:
                return u_user
            return -1.0

        # ── Kasus menolak & U_o >= U_u ───────────────────────────────
        if is_reject:
            if u_opp is None or u_opp >= u_user:
                return u_user
            return -1.0

        # ── Selain itu ───────────────────────────────────────────────
        return -1.0

    # ─────────────────────────────────────────────────────────────────
    # UTILITY-THRESHOLD — COMPUTE STEP REWARD (DLST-ANESIA, Persamaan 8)
    # ─────────────────────────────────────────────────────────────────

    def _compute_step_reward_threshold(
        self,
        step: Any,
        episode: Any,
        step_idx: int,
    ) -> float:
        """
        Hitung reward untuk satu step pada trainer utility-threshold.

        Persamaan 8 (r_ū_t):
          r_ū_t = U_u(ω_acc, t)   jika terjadi kesepakatan
                  U_u(ω^o_t, t)    jika menerima tawaran lawan
                  -1               selain itu

        Catatan penyimpanan data (lihat dlst_agent.py):
          - Saat flush di respond(): step.utility = U_u(offer lawan)  (menerima tawaran)
          - Saat flush di on_negotiation_end(): step.utility = U_u(agreement) bila ada
            agreement, atau -1 bila tidak ada agreement.
          Maka di sini, reward = step.utility yang sudah disiapkan agent
          (kecuali nilai -1 sebagai penalti gagal yang juga sudah di-encode).
        """
        mode = self.cfg.mode.lower()

        if mode == "mode1":
            u = step.utility
            if u is None:
                # Tidak ada utilitas tersimpan → penalti default
                return -1.0
            u = float(u)
            # Konvensi agent: utility == -1 menandakan kegagalan (no agreement).
            # Nilai -1 tidak di-scale (ia penalti, bukan utilitas).
            if u == -1.0:
                return -1.0
            return self._scale(u)

        # ── TAMBAHKAN MODE THRESHOLD BARU DI SINI ───────────────────
        # elif mode == "mode2":
        #     return ...
        # ────────────────────────────────────────────────────────────

        raise ValueError(
            f"[RewardShaper] mode utility_threshold tidak dikenal: '{mode}'. "
            f"Mode yang tersedia: mode1"
        )

    # ─────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────

    def compute_step_reward(
        self,
        step: Any,
        episode: Any,
        step_idx: int,
    ) -> float:
        """
        Hitung reward untuk satu step.
        Dispatch berdasarkan self.for_: "propose", "respond", atau "utility_threshold".
        """
        if self.for_ == "propose":
            return self._compute_step_reward_propose(step, episode, step_idx)
        if self.for_ == "respond":
            return self._compute_step_reward_respond(step, episode, step_idx)
        # self.for_ == "utility_threshold" (sudah divalidasi di __init__)
        return self._compute_step_reward_threshold(step, episode, step_idx)

    def compute_reward_sequence(self, episode: Any) -> List[float]:
        """
        Hitung seluruh reward sequence dari satu episode.
        Mengembalikan list reward dengan panjang = len(episode.steps).
        """
        rewards = []
        for i, step in enumerate(episode.steps):
            rewards.append(self.compute_step_reward(step, episode, i))
        return rewards

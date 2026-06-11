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
# RewardShaper
# =============================================================================
class RewardShaper:
    """
    Kelas tunggal untuk menghitung reward per-step dari data episode.

    Instansiasi:
        RewardShaper(cfg, utility_min, utility_max, for_="propose")
        RewardShaper(cfg, utility_min, utility_max, for_="respond")

    Parameter `for_` WAJIB diisi ("propose" atau "respond") dan menentukan
    jenis reward yang digunakan:
      - "propose" : menggunakan ProposeRewardConfig
      - "respond" : menggunakan RespondRewardConfig
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

    def __init__(
        self,
        cfg,
        utility_min: float = 0.0,
        utility_max: float = 1.0,
        for_: Literal["propose", "respond"] = None,
    ):
        """
        Parameters
        ----------
        cfg : ProposeRewardConfig | RespondRewardConfig
            Konfigurasi reward.
        utility_min : float
            Nilai utilitas minimum domain (untuk normalisasi).
        utility_max : float
            Nilai utilitas maksimum domain (untuk normalisasi).
        for_ : "propose" | "respond"
            Wajib diisi. Menentukan jenis reward shaper.
        """
        if for_ not in ("propose", "respond"):
            raise ValueError(
                f"[RewardShaper] Parameter 'for_' wajib diisi dengan "
                f"'propose' atau 'respond', bukan: {for_!r}"
            )
        if for_ == "propose" and not isinstance(cfg, ProposeRewardConfig):
            raise TypeError(
                "[RewardShaper] for_='propose' membutuhkan cfg bertipe ProposeRewardConfig."
            )
        if for_ == "respond" and not isinstance(cfg, RespondRewardConfig):
            raise TypeError(
                "[RewardShaper] for_='respond' membutuhkan cfg bertipe RespondRewardConfig."
            )

        self.cfg = cfg
        self.utility_min = utility_min
        self.utility_max = utility_max
        self.for_ = for_

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

    def _compute_step_reward_respond(
        self,
        step: Any,
        episode: Any,
        step_idx: int,
    ) -> float:
        """
        Hitung reward untuk satu step pada trainer respond.
        Menggunakan RespondRewardConfig.mode yang menentukan:
          (accept_variant, offer_variant, failure_penalty)
        """
        mode = self.cfg.mode.lower()
        window = self.cfg.offer_avg_window

        if mode not in self._RESPOND_MODE_TABLE:
            raise ValueError(
                f"[RewardShaper] mode respond tidak dikenal: '{mode}'. "
                f"Mode yang tersedia: {list(self._RESPOND_MODE_TABLE.keys())}"
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
        # Accept terjadi ketika: action == ACCEPT atau (step terakhir & ada agreement)
        is_accept = (action == _RESPOND_ACTION_ACCEPT) or (is_last and has_agreement)
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
        Dispatch berdasarkan self.for_: "propose" atau "respond".
        """
        if self.for_ == "propose":
            return self._compute_step_reward_propose(step, episode, step_idx)
        # self.for_ == "respond" (sudah divalidasi di __init__)
        return self._compute_step_reward_respond(step, episode, step_idx)

    def compute_reward_sequence(self, episode: Any) -> List[float]:
        """
        Hitung seluruh reward sequence dari satu episode.
        Mengembalikan list reward dengan panjang = len(episode.steps).
        """
        rewards = []
        for i, step in enumerate(episode.steps):
            rewards.append(self.compute_step_reward(step, episode, i))
        return rewards
"""
BiddingStrategy — template strategi penawaran DLST-ANESIA (Persamaan 13).

f_b(s_t, ū_t, Ω^o_t) -> offer(ω), ω ∈ Ω

Mekanisme (paper Sec. 4):
  1. Tentukan fase aktif i dari waktu t dan durasi δ.
  2. Dari taktik bidding yang AKTIF (c_{i,j} > 0.5) di fase i, pilih satu taktik
     (paper: union/pilihan taktik per fase) untuk menghasilkan bid. Bila lebih
     dari satu aktif, dipilih taktik pertama yang aktif (deterministik & rapi).
  3. Jalankan taktik tersebut untuk menghasilkan ω dari action_library.

Tactic library penawaran (T_b):
  - "boulware" : bid time-dependent Boulware (konsesi lambat di awal).
  - "pareto"   : PS(a·t + b) — pilih bid (near-)Pareto via NSGA-II + TOPSIS.
                 Di sini diaproksimasi: cari bid yang memaksimalkan
                 w1·U_u + w2·U_o dengan w1 = a·t+b, w2 = 1-w1 (TOPSIS-like).
  - "b_opp"    : manipulasi greedy tawaran terakhir lawan (ubah nilai issue
                 paling tidak relevan bagi U_u).
  - "random_above" : ω ~ U(Ω >= ū_t) — bid acak dengan utilitas >= ū_t.

# ── TAMBAHKAN TAKTIK BIDDING BARU DI SINI ──────────────────────────
#   1. Tambah method _tactic_<nama>(...) -> offer (dict)
#   2. Daftarkan di _TACTIC_FNS
#   3. Sertakan namanya di tactic_names_per_phase saat membangun spec.
"""

import random
from typing import Any, Dict, List, Optional


class BiddingStrategy:
    def __init__(self, spec, action_library, issue_names,
                 user_utility_fn, opponent_utility_fn,
                 utility_min: float, utility_max: float):
        """
        Parameters
        ----------
        spec : StrategyTemplateSpec
        action_library : list[dict]  — semua kemungkinan bid.
        issue_names : list[str]
        user_utility_fn : callable(offer) -> float (utilitas user, RAW/un-normalized)
        opponent_utility_fn : callable(offer) -> float (estimasi U_o ∈ [0,1])
        utility_min, utility_max : untuk normalisasi utilitas user.
        """
        self.spec = spec
        self.action_library = action_library
        self.issue_names = list(issue_names)
        self.user_utility_fn = user_utility_fn
        self.opponent_utility_fn = opponent_utility_fn
        self.utility_min = utility_min
        self.utility_max = utility_max

        # Pre-compute utilitas user (normalized) untuk tiap bid (efisiensi).
        self._u_user_cache = [self._norm(self.user_utility_fn(o)) for o in action_library]

    def _norm(self, u: float) -> float:
        denom = max(1e-8, self.utility_max - self.utility_min)
        return (float(u) - self.utility_min) / denom

    # ─────────────────────────────────────────────────────────────────
    # TACTIC LIBRARY (T_b) — masing-masing kembalikan offer (dict)
    # ─────────────────────────────────────────────────────────────────

    def _tactic_boulware(self, t, p, ctx) -> Dict[str, Any]:
        """
        Boulware: target utilitas tinggi di awal, konsesi makin besar mendekati t=1.
        target = 1 - t^(1/e), e kecil → konsesi lambat (Boulware). Pakai e dari p[0].
        """
        e = 0.02 + abs(float(p[0])) if len(p) > 0 else 0.1  # e > 0
        target = 1.0 - (t ** (1.0 / max(1e-3, e)))
        target = min(1.0, max(0.0, target))
        return self._closest_bid_to_target(target)

    def _tactic_pareto(self, t, p, ctx) -> Dict[str, Any]:
        """
        PS(a·t+b): pilih bid yang memaksimalkan kombinasi berbobot
        w1·U_u + w2·U_o, dengan w1 = a·t + b (clamp [0,1]) dan w2 = 1 - w1.
        Aproksimasi NSGA-II + TOPSIS (memilih bid terbaik menurut bobot waktu).
        """
        a = float(p[0]) if len(p) > 0 else -0.5
        b = float(p[1]) if len(p) > 1 else 0.8
        w1 = a * t + b
        w1 = min(1.0, max(0.0, w1))
        w2 = 1.0 - w1

        best_idx, best_score = 0, -1e18
        for idx, offer in enumerate(self.action_library):
            uu = self._u_user_cache[idx]
            uo = float(self.opponent_utility_fn(offer))
            score = w1 * uu + w2 * uo
            if score > best_score:
                best_score = score
                best_idx = idx
        return self.action_library[best_idx]

    def _tactic_b_opp(self, t, p, ctx) -> Dict[str, Any]:
        """
        Manipulasi greedy tawaran terakhir lawan: salin offer lawan, lalu ubah
        nilai issue yang paling TIDAK relevan bagi U_u menjadi nilai yang lebih
        menguntungkan agent. Aproksimasi: pilih bid di action_library yang paling
        mirip dengan offer lawan namun ber-utilitas user lebih tinggi.
        """
        opp_offer = ctx.get("opponent_last_offer", None)
        if opp_offer is None or not isinstance(opp_offer, dict):
            # fallback: bid utilitas tertinggi
            return self._best_user_bid()

        best_idx, best_score = None, -1e18
        for idx, offer in enumerate(self.action_library):
            # kemiripan = jumlah issue yang sama dengan offer lawan
            sim = sum(
                1 for k in self.issue_names
                if offer.get(k) == opp_offer.get(k)
            )
            uu = self._u_user_cache[idx]
            # utamakan mirip + utilitas user tinggi
            score = sim + uu
            if score > best_score:
                best_score = score
                best_idx = idx
        return self.action_library[best_idx if best_idx is not None else 0]

    def _tactic_random_above(self, t, p, ctx) -> Dict[str, Any]:
        """ω ~ U(Ω >= ū_t): bid acak dengan utilitas user (norm) >= ū_t."""
        u_bar = float(ctx.get("u_bar", 0.0))
        candidates = [
            offer for idx, offer in enumerate(self.action_library)
            if self._u_user_cache[idx] >= u_bar
        ]
        if len(candidates) == 0:
            return self._best_user_bid()
        return random.choice(candidates)

    @property
    def _TACTIC_FNS(self) -> Dict[str, Any]:
        # ── TAMBAHKAN TAKTIK BIDDING BARU DI SINI ───────────────────
        return {
            "boulware": self._tactic_boulware,
            "pareto": self._tactic_pareto,
            "b_opp": self._tactic_b_opp,
            "random_above": self._tactic_random_above,
        }

    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _closest_bid_to_target(self, target_norm: float) -> Dict[str, Any]:
        best_idx, best_gap = 0, float("inf")
        for idx in range(len(self.action_library)):
            gap = abs(self._u_user_cache[idx] - target_norm)
            if gap < best_gap:
                best_gap = gap
                best_idx = idx
        return self.action_library[best_idx]

    def _best_user_bid(self) -> Dict[str, Any]:
        best_idx = max(range(len(self.action_library)), key=lambda i: self._u_user_cache[i])
        return self.action_library[best_idx]

    # ─────────────────────────────────────────────────────────────────
    # GENERATE OFFER
    # ─────────────────────────────────────────────────────────────────

    def generate(self, t, action_vec, ctx) -> Dict[str, Any]:
        """
        Hasilkan ω untuk diusulkan, berdasarkan taktik bidding aktif pada fase t.
        action_vec : output actor bidding (flat).
        ctx        : dict berisi 'u_bar', 'opponent_last_offer'.
        """
        deltas, choices, params = self.spec.decode(action_vec)
        i = self.spec.active_phase(t, deltas)

        names = self.spec.tactic_names_per_phase[i]
        # Pilih taktik aktif pertama (deterministik). Jika tidak ada, fallback.
        for j, name in enumerate(names):
            if choices[i][j] > 0.5:
                fn = self._TACTIC_FNS.get(name)
                if fn is not None:
                    return fn(t, params[i][j], ctx)

        # Tidak ada taktik aktif → fallback ke bid utilitas user tertinggi
        return self._best_user_bid()

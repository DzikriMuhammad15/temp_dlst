"""
AcceptanceStrategy — template strategi penerimaan DLST-ANESIA (Persamaan 12).

f_a(s_t, ū_t, Ω^o_t) -> {accept, reject}

Mekanisme (paper Sec. 4):
  1. Tentukan fase aktif i dari waktu t dan durasi δ (output actor).
  2. Untuk tiap taktik j yang AKTIF (c_{i,j} > 0.5) di fase i, hitung ambang
     batas taktik tactic_{i,j}(...).
  3. Gabungkan ambang batas aktif via max (paper menggunakan fungsi max).
  4. Terima tawaran lawan bila U_u(ω^o_t) >= ambang batas gabungan.

Tactic library penerimaan (T_a):
  - "U_future" : U_u(ω_t) — utilitas bid yang akan diusulkan agent.
  - "quantile" : QU_{Ω^o_t}(a·t + b) — kuantil distribusi utilitas tawaran lawan.
  - "u_bar"    : ū_t — ambang batas dinamis dari DRL (threshold actor).
  - "u_fixed"  : u — ambang batas tetap (pre-defined).

# ── TAMBAHKAN TAKTIK ACCEPTANCE BARU DI SINI ───────────────────────
#   1. Tambah method _tactic_<nama>(...) -> threshold ∈ [0,1]
#   2. Daftarkan di _TACTIC_FNS
#   3. Sertakan namanya di tactic_names_per_phase saat membangun spec.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class AcceptanceStrategy:
    def __init__(self, spec, u_fixed: float = 0.6):
        """
        Parameters
        ----------
        spec : StrategyTemplateSpec
            Layout fase/taktik (untuk decode action vector).
        u_fixed : float
            Ambang batas tetap (pre-defined) untuk taktik "u_fixed".
        """
        self.spec = spec
        self.u_fixed = u_fixed

    # ─────────────────────────────────────────────────────────────────
    # TACTIC LIBRARY (T_a) — masing-masing kembalikan threshold ∈ [0,1]
    # ─────────────────────────────────────────────────────────────────

    def _tactic_U_future(self, t, p, ctx) -> float:
        """U_u(ω_t): utilitas (ternormalisasi) bid yang akan diusulkan agent."""
        return float(ctx.get("u_future_bid", 0.0))

    def _tactic_quantile(self, t, p, ctx) -> float:
        """
        QU_{Ω^o_t}(a·t + b): kuantil ke-(a·t+b) dari distribusi utilitas
        (ternormalisasi) tawaran lawan sejauh ini.
        p = [a, b].
        """
        a = float(p[0]) if len(p) > 0 else 0.0
        b = float(p[1]) if len(p) > 1 else 0.5
        q = a * t + b
        q = min(1.0, max(0.0, q))  # clamp ke [0,1]

        hist = ctx.get("opponent_util_history", [])
        if len(hist) == 0:
            return q  # tanpa data, pakai q sebagai threshold langsung
        return float(np.quantile(np.asarray(hist, dtype=np.float64), q))

    def _tactic_u_bar(self, t, p, ctx) -> float:
        """ū_t: ambang batas dinamis dari threshold actor (input eksternal)."""
        return float(ctx.get("u_bar", 0.0))

    def _tactic_u_fixed(self, t, p, ctx) -> float:
        """u: ambang batas tetap (pre-defined)."""
        return float(self.u_fixed)

    @property
    def _TACTIC_FNS(self) -> Dict[str, Any]:
        # ── TAMBAHKAN TAKTIK ACCEPTANCE BARU DI SINI ────────────────
        return {
            "U_future": self._tactic_U_future,
            "quantile": self._tactic_quantile,
            "u_bar": self._tactic_u_bar,
            "u_fixed": self._tactic_u_fixed,
        }

    # ─────────────────────────────────────────────────────────────────
    # DECIDE: accept / reject
    # ─────────────────────────────────────────────────────────────────

    def compute_threshold(self, t, action_vec, ctx) -> float:
        """
        Hitung ambang batas penerimaan gabungan pada waktu t.
        action_vec : output actor acceptance (flat).
        ctx        : dict berisi 'u_bar', 'u_future_bid', 'opponent_util_history'.
        """
        deltas, choices, params = self.spec.decode(action_vec)
        i = self.spec.active_phase(t, deltas)

        names = self.spec.tactic_names_per_phase[i]
        thresholds = []
        for j, name in enumerate(names):
            if choices[i][j] <= 0.5:
                continue  # taktik tidak aktif (c_{i,j} == false)
            fn = self._TACTIC_FNS.get(name)
            if fn is None:
                continue
            thresholds.append(fn(t, params[i][j], ctx))

        if len(thresholds) == 0:
            # Tidak ada taktik aktif → fallback ke ambang batas dinamis
            return float(ctx.get("u_bar", 0.0))
        return float(max(thresholds))  # gabungan via max (paper)

    def decide(self, t, action_vec, opponent_offer_util_norm, ctx) -> bool:
        """
        Return True jika agent menerima tawaran lawan.
        opponent_offer_util_norm : U_u(ω^o_t) ternormalisasi ∈ [0,1].
        """
        threshold = self.compute_threshold(t, action_vec, ctx)
        return float(opponent_offer_util_norm) >= threshold

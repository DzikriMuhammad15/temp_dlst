"""
StrategyTemplateSpec — definisi layout output actor strategy (Persamaan 21).

Actor strategy menghasilkan vektor flat: [δ_1..δ_n, c_{1,1}..c_{n,n_i}, p-flat...].
Spec ini menyimpan:
  - n_phases       : jumlah fase (n)
  - tactics_per_phase : list[int], n_i untuk tiap fase (jumlah taktik di fase i)
  - params_per_tactic : jumlah nilai numerik p untuk tiap taktik (default 2: a,b)

Sehingga:
  n_delta  = n_phases
  n_choice = Σ_i n_i
  n_param  = Σ_i n_i * params_per_tactic
  action_dim = n_delta + n_choice + n_param

Layout flat:
  [ δ_1, ..., δ_n,
    c_{1,1}, ..., c_{1,n_1}, c_{2,1}, ..., c_{n,n_n},
    p_{1,1}(a,b), p_{1,2}(a,b), ..., p_{n,n_n}(a,b) ]

Catatan: tactic_names_per_phase memberi tahu taktik MANA (dari tactic library)
yang menempati slot j di fase i, untuk acceptance dan bidding terpisah.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class StrategyTemplateSpec:
    n_phases: int
    tactics_per_phase: List[int]          # n_i untuk tiap fase
    tactic_names_per_phase: List[List[str]]  # nama taktik per slot per fase
    params_per_tactic: int = 2            # jumlah nilai p per taktik (a, b)

    def __post_init__(self):
        assert self.n_phases == len(self.tactics_per_phase), (
            "tactics_per_phase harus sepanjang n_phases"
        )
        assert self.n_phases == len(self.tactic_names_per_phase), (
            "tactic_names_per_phase harus sepanjang n_phases"
        )
        for i in range(self.n_phases):
            assert self.tactics_per_phase[i] == len(self.tactic_names_per_phase[i]), (
                f"Jumlah taktik fase {i} tidak konsisten"
            )

    @property
    def n_delta(self) -> int:
        return self.n_phases

    @property
    def n_choice(self) -> int:
        return sum(self.tactics_per_phase)

    @property
    def n_param(self) -> int:
        return sum(self.tactics_per_phase) * self.params_per_tactic

    @property
    def action_dim(self) -> int:
        return self.n_delta + self.n_choice + self.n_param

    @property
    def split_sizes(self):
        return (self.n_delta, self.n_choice, self.n_param)

    # ─────────────────────────────────────────────────────────────────
    # DECODE: pisahkan flat action vector → (deltas, choices, params)
    # ─────────────────────────────────────────────────────────────────

    def decode(self, action_vec):
        """
        action_vec : array-like panjang action_dim.
        Return:
          deltas  : list[float] panjang n_phases
          choices : list[list[float]] (choices[i][j] ∈ (0,1))
          params  : list[list[list[float]]] (params[i][j] = [a, b, ...])
        """
        v = list(action_vec)
        assert len(v) == self.action_dim, (
            f"action_vec len={len(v)} != action_dim={self.action_dim}"
        )

        idx = 0
        deltas = v[idx:idx + self.n_delta]
        idx += self.n_delta

        choices = []
        for i in range(self.n_phases):
            ni = self.tactics_per_phase[i]
            choices.append(v[idx:idx + ni])
            idx += ni

        params = []
        for i in range(self.n_phases):
            ni = self.tactics_per_phase[i]
            phase_params = []
            for _ in range(ni):
                phase_params.append(v[idx:idx + self.params_per_tactic])
                idx += self.params_per_tactic
            params.append(phase_params)

        return deltas, choices, params

    # ─────────────────────────────────────────────────────────────────
    # PHASE BOUNDARIES dari deltas
    # ─────────────────────────────────────────────────────────────────

    def phase_bounds(self, deltas):
        """
        Hitung batas fase [t_i, t_{i+1}) dari durasi δ_i.
        δ dinormalisasi agar Σ δ = 1 (paper: t_1=0, t_{n+1}=1).
        Return list of (t_start, t_end) panjang n_phases.
        """
        s = sum(deltas)
        if s <= 1e-8:
            # fallback: bagi rata
            norm = [1.0 / self.n_phases] * self.n_phases
        else:
            norm = [d / s for d in deltas]

        bounds = []
        t = 0.0
        for i in range(self.n_phases):
            t_start = t
            t_end = t + norm[i]
            bounds.append((t_start, t_end))
            t = t_end
        # pastikan fase terakhir menutup di 1.0
        if bounds:
            ts, _ = bounds[-1]
            bounds[-1] = (ts, 1.0)
        return bounds

    def active_phase(self, t, deltas):
        """
        Tentukan index fase aktif untuk waktu t ∈ [0,1].
        Sesuai paper: jika t sudah melewati suatu fase, langsung ke fase
        yang mengandung t. Return index fase (0-based).
        """
        bounds = self.phase_bounds(deltas)
        for i, (ts, te) in enumerate(bounds):
            # [t_i, t_{i+1}); fase terakhir inklusif di ujung
            if i < self.n_phases - 1:
                if ts <= t < te:
                    return i
            else:
                if ts <= t <= te:
                    return i
        # t di luar [0,1] (mis. t>1) → fase terakhir
        return self.n_phases - 1

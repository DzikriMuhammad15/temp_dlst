
from typing import List

import numpy as np


class BayesianMixer:
    def __init__(
        self,
        M: int,
        decay: float = 0.9,
        horizon: int = 10,
        temperature: float = 1.0,
    ):
        self.M = M
        self.decay = decay
        self.horizon = horizon
        self.temperature = temperature

        # Bobot akhir setelah softer-softmax
        self.alpha: np.ndarray = np.ones(M, dtype=np.float64) / M

        # Ψ_m = decayed moving average dari p(m | a^o)
        self._psi: np.ndarray = np.ones(M, dtype=np.float64) / M

        # Prior p(m) = moving average dari p(m | a^o)
        self._prior: np.ndarray = np.ones(M, dtype=np.float64) / M

        # Ringkasan history p(m | a^o) selama horizon H
        self._history: List[np.ndarray] = []

        # Timestep counter
        self._t: int = 0

    # ------------------------------------------------------------------
    # BAYESIAN UPDATE  (Eq. 7)
    # ------------------------------------------------------------------

    def update(self, iop_probs_for_action: np.ndarray) -> None:
        """
        Update α berdasarkan probabilitas p(a_o | m) dari setiap IOP
        untuk action yang benar-benar dilakukan lawan.

        Args:
            iop_probs_for_action: array shape (M,) — setiap elemen m adalah
                π̃^o_m(a_o | s; φ_m) untuk action riil yang diamati.
        """
        # p(m | a^o) = π̃^o_m(a_o) * p(m) / Σ_i [π̃^o_i(a_o) * p(i)]   (Eq. 7)
        numerators = iop_probs_for_action * self._prior
        denom = numerators.sum()
        if denom < 1e-10:
            posterior = np.ones(self.M, dtype=np.float64) / self.M
        else:
            posterior = numerators / denom

        # Append ke history
        self._history.append(posterior.copy())
        if len(self._history) > self.horizon:
            self._history.pop(0)

        self._t += 1

        # Update prior p(m) = moving average dari p(m|a^o) selama H langkah
        if self._history:
            self._prior = np.mean(np.array(self._history), axis=0)

        # Update Ψ_m (decayed moving average) — Appendix B
        # Ψ^t_m = Σ_{l=t-H}^{t-1} λ^{t-l} · p(m | a^o_l)
        psi = np.zeros(self.M, dtype=np.float64)
        for j, hist_posterior in enumerate(reversed(self._history)):
            # j=0 adalah yang terbaru, j=H-1 yang paling lama
            psi += (self.decay ** (j + 1)) * hist_posterior
        self._psi = psi

        # Perbarui α dengan softer-softmax (Eq. 8)
        self.alpha = self._softer_softmax(self._psi)

    def _softer_softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softer-softmax dengan temperature > 1 sesuai referensi Hinton et al. [12].
        temperature >= 1 menghasilkan distribusi yang lebih merata dari softmax biasa.
        """
        scaled = x / max(self.temperature, 1e-8)
        exp_x = np.exp(scaled - scaled.max())  # stabilisasi numerik
        return exp_x / (exp_x.sum() + 1e-10)

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------

    def get_mixed_probs(self, iop_probs_list: List[np.ndarray]) -> np.ndarray:
        """
        Hitung distribusi kebijakan campuran:
          π̃^o_mix(·|s) = Σ_m α_m · π̃^o_m(·|s; φ_m)   (Eq. 6)

        Args:
            iop_probs_list: list M array, setiap array shape (action_dim,)
                            — distribusi probabilitas dari setiap IOP level.
        Returns:
            mixed distribusi probabilitas shape (action_dim,)
        """
        mixed = np.zeros_like(iop_probs_list[0], dtype=np.float64)
        for m, probs in enumerate(iop_probs_list):
            mixed += float(self.alpha[m]) * np.array(probs, dtype=np.float64)
        total = mixed.sum()
        if total < 1e-10:
            return np.ones(len(mixed)) / len(mixed)
        return (mixed / total).astype(np.float32)

    def get_mixed_action_idx(self, iop_probs_list: List[np.ndarray]) -> int:
        """
        Sample satu action index dari distribusi campuran π̃^o_mix.
        Digunakan oleh rollout engine untuk simulasi aksi lawan.
        """
        mixed_probs = self.get_mixed_probs(iop_probs_list)
        mixed_probs = mixed_probs / mixed_probs.sum()
        return int(np.random.choice(len(mixed_probs), p=mixed_probs))

    def get_alpha(self) -> np.ndarray:
        """Kembalikan salinan bobot α saat ini."""
        return self.alpha.copy()

    def reset(self) -> None:
        """Reset state untuk negosiasi baru."""
        self.alpha = np.ones(self.M, dtype=np.float64) / self.M
        self._psi = np.ones(self.M, dtype=np.float64) / self.M
        self._prior = np.ones(self.M, dtype=np.float64) / self.M
        self._history.clear()
        self._t = 0

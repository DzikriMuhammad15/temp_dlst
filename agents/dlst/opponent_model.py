"""
Opponent utility approximation (frequency model) untuk DLST-ANESIA.

Implementasi berdasarkan distribution-based frequency model (Tunali et al., 2017,
seperti dirujuk paper DLST-ANESIA, Sec. 5.4). Model ini mengestimasi:
  - bobot tiap issue (issue weights) lawan
  - nilai evaluasi tiap value pada tiap issue (value evaluations) lawan
hanya dari riwayat tawaran lawan (Ω^o_t), tanpa tahu preferensi asli lawan.

Catatan penting tentang peran di project:
  Object ini MENYATU dengan reward shaper respond (lihat reward_shaper.py).
  Reward respond (r_acc, Persamaan 10) butuh utilitas lawan U_o(ω). Karena agent
  tidak tahu U_o asli, kita aproksimasi via frequency model ini. Ketika agent
  butuh estimasi utilitas lawan (mis. untuk Pareto bidding / TOPSIS), agent
  mengambilnya dari trainer respond -> reward_shaper -> opponent_model ini.
"""

from typing import Any, Dict, List, Optional


class FrequencyOpponentModel:
    """
    Estimasi U_o (utilitas lawan) berbasis frekuensi kemunculan nilai issue.

    Mekanisme:
      - Tiap kali lawan menawarkan ω, kita catat nilai tiap issue di ω.
      - Value evaluation e_i(v) diestimasi dari frekuensi: nilai yang sering
        ditawarkan lawan dianggap lebih disukai lawan (rank-based normalization).
      - Issue weight w_i diestimasi dari seberapa "stabil" (tidak berubah-ubah)
        nilai issue tersebut pada riwayat tawaran lawan: issue yang nilainya
        konsisten/dominan dianggap lebih penting bagi lawan.
      - U_o(ω) = Σ_i w_i * e_i(ω[i]), dinormalisasi ke [0, 1].
    """

    def __init__(self, issue_names: List[str], value_lists: Dict[str, List[Any]]):
        """
        Parameters
        ----------
        issue_names : list[str]
            Nama-nama issue (urutan tetap).
        value_lists : dict[str, list]
            Untuk tiap issue, daftar kemungkinan nilainya.
        """
        self.issue_names = list(issue_names)
        self.value_lists = {k: list(v) for k, v in value_lists.items()}

        # Frekuensi kemunculan tiap (issue, value) pada tawaran lawan.
        # freq[issue][value] = jumlah kemunculan
        self.freq: Dict[str, Dict[Any, int]] = {
            issue: {val: 0 for val in self.value_lists.get(issue, [])}
            for issue in self.issue_names
        }
        self.total_bids = 0

        # print(f"self.freq: {self.freq}")
        # print(f"self.value_lists: {self.value_lists}")

    # ─────────────────────────────────────────────────────────────────
    # UPDATE
    # ─────────────────────────────────────────────────────────────────

    def update(self, opponent_offer: Optional[Dict[str, Any]]) -> None:
        """
        Catat satu tawaran lawan ke dalam statistik frekuensi.
        offer berupa dict {issue_name: value} (format action_library).
        """
        if opponent_offer is None:
            return
        if not isinstance(opponent_offer, dict):
            return

        for issue in self.issue_names:
            if issue not in opponent_offer:
                continue
            val = opponent_offer[issue]
            if issue not in self.freq:
                self.freq[issue] = {}
            if val not in self.freq[issue]:
                self.freq[issue][val] = 0
            self.freq[issue][val] += 1

        self.total_bids += 1
        print(f"total_bid: {self.total_bids}")
        print(f"self. freq: {self.freq}")

    # ─────────────────────────────────────────────────────────────────
    # ESTIMASI KOMPONEN
    # ─────────────────────────────────────────────────────────────────

    def _value_evaluations(self, issue: str) -> Dict[Any, float]:
        """
        Estimasi e_i(v) untuk tiap value v pada issue tertentu (rank-based).
        Value dengan frekuensi tertinggi mendapat nilai evaluasi mendekati 1,
        terendah mendekati 1/k. Mengembalikan map value -> eval ∈ (0, 1].
        """
        counts = self.freq.get(issue, {})
        values = self.value_lists.get(issue, list(counts.keys()))
        k = len(values)
        if k == 0:
            return {}

        # Urutkan value berdasar frekuensi (ascending), beri rank 1..k.
        # eval = rank / k  → value paling sering = 1.0
        sorted_vals = sorted(values, key=lambda v: counts.get(v, 0))
        evals: Dict[Any, float] = {}
        for rank, v in enumerate(sorted_vals, start=1):
            evals[v] = rank / k
        return evals

    def _issue_weights(self) -> Dict[str, float]:
        """
        Estimasi w_i: issue dianggap lebih penting bila lawan konsisten
        menawarkan nilai yang sama (konsentrasi frekuensi tinggi).
        Diukur via sum-of-squared relative frequencies (Herfindahl-like),
        lalu dinormalisasi agar Σ w_i = 1.
        """
        raw: Dict[str, float] = {}
        for issue in self.issue_names:
            counts = self.freq.get(issue, {})
            total = sum(counts.values())
            if total <= 0:
                raw[issue] = 0.0
                continue
            concentration = sum((c / total) ** 2 for c in counts.values())
            raw[issue] = concentration

        s = sum(raw.values())
        if s <= 1e-12:
            # Belum ada data: bobot seragam
            n = len(self.issue_names)
            return {issue: 1.0 / n for issue in self.issue_names} if n > 0 else {}
        return {issue: raw[issue] / s for issue in self.issue_names}

    # ─────────────────────────────────────────────────────────────────
    # ESTIMASI UTILITAS LAWAN
    # ─────────────────────────────────────────────────────────────────

    def estimate_utility(self, offer: Optional[Dict[str, Any]]) -> float:
        """
        Estimasi U_o(offer) ∈ [0, 1] berdasarkan model frekuensi saat ini.
        Jika belum ada data sama sekali, kembalikan 0.5 (netral).
        """
        if offer is None or not isinstance(offer, dict):
            return 0.0
        if self.total_bids == 0:
            return 0.5

        weights = self._issue_weights()
        u = 0.0
        for issue in self.issue_names:
            if issue not in offer:
                continue
            evals = self._value_evaluations(issue)
            e = evals.get(offer[issue], 0.0)
            u += weights.get(issue, 0.0) * e
        # Σ weights = 1 dan e ∈ (0,1] → u ∈ [0,1]
        return float(min(1.0, max(0.0, u)))

    def reset(self) -> None:
        """Reset statistik (mis. di awal episode baru jika diinginkan)."""
        self.freq = {
            issue: {val: 0 for val in self.value_lists.get(issue, [])}
            for issue in self.issue_names
        }
        self.total_bids = 0

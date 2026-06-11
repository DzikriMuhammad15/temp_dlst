from typing import Any, Dict, Optional, Tuple

from negmas import SAOMechanism
from negmas.sao.common import SAOState


# ─────────────────────────────────────────────────────────────────────────────
# CustomStartSAOMechanism
# ─────────────────────────────────────────────────────────────────────────────

class CustomStartSAOMechanism(SAOMechanism):
    def __init__(
        self,
        *args,
        initial_offer: Optional[Dict[str, Any]] = None,
        first_actor: str = "a",
        start_with: str = "propose",
        **kwargs,
    ):
        if initial_offer is not None and "initial_state" not in kwargs:
            kwargs["initial_state"] = SAOState(current_offer=initial_offer)

        super().__init__(*args, **kwargs)

        self._custom_initial_offer = initial_offer
        self._custom_first_actor   = first_actor
        self._custom_start_with    = start_with

    def on_negotiation_start(self) -> bool:
        """
        Dipanggil oleh base Mechanism.run() SETELAH semua .add() selesai
        dan SEBELUM step pertama.
        """
        result = super().on_negotiation_start()

        first_idx  = 0 if self._custom_first_actor == "a" else 1
        second_idx = 1 - first_idx
        n          = len(self.negotiators)

        if self._custom_start_with == "propose":
            # ── first_actor propose duluan ────────────────────────────
            # initial_offer (jika ada) dianggap milik first_actor.
            if self._custom_initial_offer is not None:
                proposer = self.negotiators[first_idx]
                self._current_state.current_offer    = self._custom_initial_offer
                self._current_state.current_proposer = proposer.id
                self._current_proposer               = proposer

        else:  # "respond"
            # ── first_actor respond duluan terhadap initial_offer ─────
            # initial_offer "milik" second_actor (dia proposer-nya),
            # sehingga first_actor bukan _current_proposer → dia respond.
            if self._custom_initial_offer is not None:
                proposer = self.negotiators[second_idx]
                self._current_state.current_offer    = self._custom_initial_offer
                self._current_state.current_proposer = proposer.id
                self._current_proposer               = proposer

        # Atur siapa yang dipanggil pertama.
        # Rumus di __call__:
        #   ordered_indices[0] = (_last_checked_negotiator + 1) % n
        # Sehingga untuk ordered_indices[0] = first_idx:
        #   _last_checked_negotiator = (first_idx - 1) % n
        self._last_checked_negotiator = (first_idx - 1) % n

        return result


# ─────────────────────────────────────────────────────────────────────────────
# NegMASBilateralEnv
# ─────────────────────────────────────────────────────────────────────────────

class NegMASBilateralEnv:
    def __init__(self, domain_spec, max_steps: int = 20):
        self.domain_spec = domain_spec
        self.max_steps   = max_steps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_agent(self, agent, skip_negotiation_state: bool = False):
        for attr in [
            "_nmi", "_mechanism", "_negotiation_result",
            "_current_offer", "_last_offer", "_offered",
            "__negotiator_id__", "_Negotiator__parent",
        ]:
            if hasattr(agent, attr):
                try:
                    setattr(agent, attr, None)
                except Exception:
                    pass

        if not skip_negotiation_state:
            agent.memory_propose             = []
            agent.memory_respond             = []
            agent.current_state_data_propose = None
            agent.current_state_data_respond = None
            agent.last_episode_memory        = []
            agent.prev_offer_utility         = None
            agent.last_received_offer        = None
            agent.last_proposed_offer        = None

    def _extract_agreement(self, mechanism, returned_state):
        mech_agreement = getattr(mechanism, "agreement", None)
        if mech_agreement is not None:
            return mech_agreement
        if hasattr(returned_state, "agreement"):
            return getattr(returned_state, "agreement")
        if isinstance(returned_state, dict):
            return returned_state
        return None

    def _extract_trace(self, mechanism):
        if hasattr(mechanism, "trace"):
            return mechanism.trace
        if hasattr(mechanism, "full_trace"):
            return mechanism.full_trace
        if hasattr(mechanism, "extended_trace"):
            return mechanism.extended_trace
        return None

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(
        self,
        agent_a,
        agent_b,
        initial_offer=None,
        first_actor: str = "a",
        start_with: str = "propose",
        skip_agent_reset: bool = False,
    ) -> tuple:
        if first_actor not in ("a", "b"):
            raise ValueError("first_actor harus 'a' atau 'b'.")
        if start_with not in ("propose", "respond"):
            raise ValueError("start_with harus 'propose' atau 'respond'.")
        if start_with == "respond" and initial_offer is None:
            raise ValueError(
                "start_with='respond' membutuhkan initial_offer yang tidak None."
            )
        if initial_offer is not None and not isinstance(initial_offer, dict):
            raise TypeError(
                f"initial_offer harus berupa dict (format action_library), "
                f"bukan {type(initial_offer).__name__}."
            )

        # Reset agent — untuk agent aktif di episode lain, hanya reset
        # state NegMAS internal (skip_negotiation_state=True)
        self._reset_agent(agent_a, skip_negotiation_state=skip_agent_reset)
        self._reset_agent(agent_b, skip_negotiation_state=False)  # opponent selalu full reset

        mechanism = CustomStartSAOMechanism(
            issues=self.domain_spec.issues,
            n_steps=self.max_steps,
            initial_offer=initial_offer,
            first_actor=first_actor,
            start_with=start_with,
        )

        mechanism.add(agent_a)
        mechanism.add(agent_b)

        returned  = mechanism.run()
        agreement = self._extract_agreement(mechanism, returned)
        trace     = self._extract_trace(mechanism)

        summary = {
            "agreement"    : agreement,
            "trace"        : trace,
            "n_steps"      : self.max_steps,
            "domain_name"  : self.domain_spec.domain_name,
            "initial_offer": initial_offer,
            "first_actor"  : first_actor,
            "start_with"   : start_with,
        }
        return agreement, summary
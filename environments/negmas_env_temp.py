from typing import Any, Dict, Tuple

from negmas import SAOMechanism


class NegMASBilateralEnv:
    def __init__(self, domain_spec, max_steps: int = 20):
        self.domain_spec = domain_spec
        self.max_steps = max_steps

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

    def run(self, agent_a, agent_b) -> Tuple[Any, Dict[str, Any]]:
        mechanism = SAOMechanism(issues=self.domain_spec.issues, n_steps=self.max_steps)

        for agent in (agent_a, agent_b):
            # Reset semua atribut NegMAS internal
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

            # Reset state kustom kita
            agent.memory = []
            agent.last_episode_memory = []
            agent.prev_offer_utility = None
            agent.last_received_offer = None
            agent.last_proposed_offer = None

        mechanism.add(agent_a)
        mechanism.add(agent_b)

        returned = mechanism.run()

        agreement = self._extract_agreement(mechanism, returned)
        trace = self._extract_trace(mechanism)


        summary = {
            "agreement": agreement,
            "trace": trace,
            "n_steps": self.max_steps,
            "domain_name": self.domain_spec.domain_name,
        }
        return agreement, summary



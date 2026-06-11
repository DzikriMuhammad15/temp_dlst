from copy import deepcopy
from typing import Any, Dict, List, Optional
import numpy as np

from negmas import SAONegotiator


from data_type.episode import Episode
from data_type.step import Step

from abc import ABC, abstractmethod

# interface
class NegosiatorFunctionality(ABC):
    @abstractmethod
    def attach_trainer(self, dict_of_trainer):
        pass

    @abstractmethod
    def build_state(self, state):
        pass

    @abstractmethod
    def on_negotiation_start(self, state):
        pass

    @abstractmethod
    def on_negotiation_end(self, state):
        pass

    @abstractmethod
    def respond(self, state, source=None):
        pass

    @abstractmethod
    def propose(self, state):
        pass

    @abstractmethod
    def build_episode_payload_propose(self, state):
        pass

    @abstractmethod
    def build_episode_payload_respond(self, state):
        pass

    @abstractmethod
    def reset_episode(self):
        pass

    @abstractmethod
    def sample_action_propose(self, obs):
        pass

    @abstractmethod
    def sample_action_respond(self, obs):
        pass

    @abstractmethod
    def decode_action(self, action_id):
        pass

    @abstractmethod
    def _offer_to_action_idx(self, offer):
        pass

    @abstractmethod
    def _offer_utility(self, offer):
        pass

    @abstractmethod
    def _mask_logits(self, logits, allow_accept):
        pass




class BaseNegotiator(SAONegotiator, NegosiatorFunctionality):
    """
    Base class untuk semua negotiator.

    Menyimpan:
    - memory_propose: langkah propose per episode
    - memory_respond: langkah respond per episode
    - snapshot episode terakhir
    - ringkasan episode
    """

    def __init__(self, name: str, ufun, domain_spec: Any, trainers=None, **kwargs):
        super().__init__(name=name, ufun=ufun, **kwargs)
        self.name = name
        self.domain_spec = domain_spec

        # Dipisah: memory propose dan respond
        self.memory_propose: List[Step] = []
        self.memory_respond: List[Step] = []

        self.trainers = trainers if trainers is not None else {}
        self.ufun = ufun
        self.current_state_data_propose = None
        self.current_state_data_respond = None


    def _domain_attr(self, key: str, default=None):
        if isinstance(self.domain_spec, dict):
            return self.domain_spec.get(key, default)
        return getattr(self.domain_spec, key, default)

    def on_negotiation_start(self, state):
        """
        Reset buffer episode saat negotiation dimulai.
        """
        self.memory_propose = []
        self.memory_respond = []

    def record_step_propose(
        self,
        state=None,
        action=None,
        utility=None,
        value=None,
        q_value=None,
        log_prob=None,
        next_state=None
    ):
        """
        Simpan satu transition propose agar bisa dijadikan dataset.
        """
        self.memory_propose.append(Step(
            state=state,
            action=action,
            utility=utility,
            value=value,
            q_value=q_value,
            log_prob=log_prob,
            next_state=next_state
        ))

    def record_step_respond(
        self,
        state=None,
        action=None,
        utility=None,
        value=None,
        q_value=None,
        log_prob=None,
        next_state=None
    ):
        """
        Simpan satu transition respond agar bisa dijadikan dataset.
        """
        self.memory_respond.append(Step(
            state=state,
            action=action,
            utility=utility,
            value=value,
            q_value=q_value,
            log_prob=log_prob,
            next_state=next_state
        ))

    def on_negotiation_end(self, state):
        """
        Default: simpan episode lalu kosongkan memory.
        """
        pass

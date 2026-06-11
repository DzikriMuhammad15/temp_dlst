from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from data_type.step import Step

@dataclass
class Episode:
    """
    Representasi satu episode dalam trajectory.
    """
    agent_name: Any
    terminal_utility: Any
    agreement: Any
    states: Any
    actions: Any
    log_probs: Any
    values: Any
    q_values: Any
    utilities: Any
    steps: List[Step]
    meta: Any = None
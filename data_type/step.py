from dataclasses import dataclass
from typing import Any

@dataclass
class Step:
    """
    Representasi satu langkah dalam trajectory.
    """
    state: Any = None
    action: Any = None
    utility: Any = None
    value: Any = None
    q_value: Any = None
    log_prob: Any = None
    next_state: Any = None

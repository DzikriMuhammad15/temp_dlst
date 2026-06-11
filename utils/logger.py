from datetime import datetime
from typing import Any, Dict


class SimpleLogger:
    """
    Logger ringan agar output eksperimen tetap rapi.
    """

    def __init__(self, prefix: str = "RUN"):
        self.prefix = prefix

    def log(self, message: str):
        now = datetime.now().strftime("%H:%M:%S")
        # print(f"[{now}][{self.prefix}] {message}")

    def log_dict(self, title: str, data: Dict[str, Any]):
        self.log(title)
        for k, v in data.items():
            self.log(f"  - {k}: {v}")
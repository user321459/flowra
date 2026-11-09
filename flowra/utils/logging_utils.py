"""Experiment logging: human-readable ``run.log`` plus machine-readable ``metrics.jsonl``."""

import json
import logging
import os
import platform
import sys
import time
from typing import Any, Dict, Optional

import torch


class ExperimentLogger:
    def __init__(self, log_dir: Optional[str] = None, name: str = "flowra", console: bool = True):
        self.log_dir = log_dir
        self.logger = logging.getLogger(f"{name}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        if console:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            self.logger.addHandler(sh)
        self._jsonl = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            fh = logging.FileHandler(os.path.join(log_dir, "run.log"), mode="w", encoding="utf-8")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)
            self._jsonl = open(os.path.join(log_dir, "metrics.jsonl"), "w", encoding="utf-8")
        self._t0 = time.time()

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def log(self, event: str, payload: Dict[str, Any], message: Optional[str] = None) -> None:
        record = {"event": event, "time": round(time.time() - self._t0, 3), **payload}
        if self._jsonl:
            self._jsonl.write(json.dumps(record, default=_json_default) + "\n")
            self._jsonl.flush()
        if message:
            self.info(message)

    def log_environment(self) -> None:
        env = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "cpu",
            "platform": platform.platform(),
        }
        self.log("environment", env, "environment: " + ", ".join(f"{k}={v}" for k, v in env.items()))

    def save_json(self, filename: str, data: Any) -> None:
        if self.log_dir:
            with open(os.path.join(self.log_dir, filename), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=_json_default)

    def close(self) -> None:
        if self._jsonl:
            self._jsonl.close()
            self._jsonl = None
        for h in list(self.logger.handlers):
            h.close()
            self.logger.removeHandler(h)


def _json_default(o: Any):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)

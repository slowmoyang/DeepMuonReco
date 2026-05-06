import logging
from typing import Any
from pathlib import Path
import torch

__all__ = ["ModelCheckpoint"]

_logger = logging.getLogger(__name__)


class ModelCheckpoint:
    def __init__(
        self,
        object_dict: dict[str, Any],
        mode: str,
        output_dir: Path,
        metric: str = "metric",
        save_last: bool = False,
    ) -> None:
        """
        Args:
            object_dict: A dictionary of objects to save. The keys will be used as the names in the checkpoint.
            metric: The name of the metric to monitor (e.g., "val_loss", "accuracy").
            mode: One of "min" or "max". In "min" mode, the checkpoint will be saved when the monitored metric decreases. In "max" mode, it will be saved when the monitored metric increases.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.object_dict = object_dict
        self.metric = metric
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_last = save_last

        self.best_path = self.output_dir / "best.pt"
        if self.save_last:
            self.last_path = self.output_dir / "last.pt"
        else:
            self.last_path = None

    def _is_improved(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < self.best_metric
        else:
            return metric > self.best_metric

    def _save(self, metric: float, path: Path) -> None:
        checkpoint = {
            self.metric: metric,
        }
        for name, obj in self.object_dict.items():
            if hasattr(obj, "state_dict"):
                checkpoint[name] = obj.state_dict()
            else:
                checkpoint[name] = obj

        torch.save(checkpoint, path)

    def step(self, metric: float | dict[str, Any]) -> None:
        if isinstance(metric, dict):
            metric = metric[self.metric]

        if not isinstance(metric, (int, float)):
            raise ValueError(f"Metric must be a number, got {type(metric)}")

        if self.save_last:
            _logger.debug(
                f"Saving last checkpoint to {self.last_path} with metric {self.metric}={metric:.4f}."
            )
            self._save(metric, self.last_path)

        if not self._is_improved(metric):
            return

        _logger.info(f"Metric improved from {self.best_metric:.4f} to {metric:.4f}.")
        self.best_metric = metric

        _logger.info(
            f"Saving best checkpoint to {self.best_path} with metric {self.metric}={metric:.4f}."
        )
        self._save(metric, self.best_path)

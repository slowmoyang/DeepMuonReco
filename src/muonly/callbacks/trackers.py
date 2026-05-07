import os
from pathlib import Path
import logging
from dataclasses import dataclass
import abc
import psutil
import torch
import pandas as pd
import humanize


__all__ = ["MemoryTracker", "CUDAMemoryTracker"]



@dataclass(frozen=True)
class MemoryUsage:
    tag: str
    rss_bytes: int


class Tracker(abc.ABC):

    def __init__(
        self,
        output_dir: Path,
        log_file_name: str,
    ) -> None:
        self.output_dir = output_dir
        self.log_file_name = log_file_name

        self.logger = logging.getLogger(self.__class__.__name__)
        self.log = []
        self.track("tracking_start")

    @abc.abstractmethod
    def track(self, tag: str):
        ...

    def __del__(self):
        self.track("tracking_end")
        self.write(self.output_dir / self.log_file_name)

    def write(self, path: Path):
        df = pd.DataFrame(self.log)
        df.to_csv(path, index=False)



class MemoryTracker(Tracker):
    def __init__(
        self,
        output_dir: Path,
        log_file_name: str = "memory.csv",
    ) -> None:
        """
        """
        self.process = psutil.Process(os.getpid())
        super().__init__(output_dir=output_dir, log_file_name=log_file_name)

    def track(self, tag: str):
        mem = self.process.memory_info().rss
        self.log.append(dict(tag=tag, rss_bytes=mem))
        self.logger.info(f"{tag}: {humanize.naturalsize(mem, binary=True)}")


class CUDAMemoryTracker(Tracker):
    def __init__(
        self,
        device: torch.device,
        output_dir: Path,
        log_file_name: str = "cuda-memory.csv",
    ):
        self.device = device

        if self.device.type != "cuda":
            self.logger.warning(
                "CUDAMemoryTracker initialized on non-CUDA device. Memory tracking will be disabled."
            )

        super().__init__(
            output_dir=output_dir,
            log_file_name=log_file_name,
        )

    def __del__(self):
        if self.device.type != "cuda":
            return
        self.track("final")
        self.write(self.output_dir / "cuda-memory.csv")
        self.summarize(self.output_dir / "cuda-memory-summary.txt")

    def track(self, tag: str):
        if self.device.type != "cuda":
            return

        mem = torch.cuda.memory_allocated(device=self.device)
        max_mem = torch.cuda.max_memory_allocated(device=self.device)

        self.log.append(dict(tag=tag, allocated_bytes=mem, max_allocated_bytes=max_mem))

        self.logger.info(
            f"{tag}: {humanize.naturalsize(mem, binary=True)} "
            f"(Peak: {humanize.naturalsize(max_mem, binary=True)})"
        )

    def __del__(self):
        if self.device.type != "cuda":
            return
        super().__del__()
        self.summarize(self.output_dir / "cuda-memory-summary.txt")

    def summarize(self, path: Path):
        if self.device.type != "cuda":
            return

        device_properties = torch.cuda.get_device_properties(self.device)
        summary = torch.cuda.memory_summary(device=self.device, abbreviated=False)

        log = (
            f"Device name: {device_properties.name}\n"
            f"Device UUID: {device_properties.uuid}\n"
            f"{summary}"
        )

        self.logger.debug(f"Writing CUDA memory summary to {path}...")
        with path.open("w") as file:
            file.write(log)

import os
from pathlib import Path
import logging
import aim
import psutil
import torch
import pandas as pd
import humanize


__al__ = ["MemoryTracker", "CUDAMemoryTracker"]


class MemoryTracker:

    def __init__(
        self,
        output_dir: Path | None = None,
    ) -> None:
        self.output_dir = output_dir

        self.process = psutil.Process(os.getpid())
        self.logger = logging.getLogger(self.__class__.__name__)

        self.log: list[tuple[str, int]] = []

        self.track("initial")

    def __del__(self):
        if self.output_dir is None:
            return
        self.track("final")
        self.write(self.output_dir / "memory.csv")

    def track(self, tag: str):
        mem = self.process.memory_info().rss

        self.log.append((tag, mem))

        self.logger.info(f"{tag}: {humanize.naturalsize(mem, binary=True)}")

    def write(self, path: Path):
        df = pd.DataFrame(self.log, columns=["tag", "rss_bytes"])
        self.logger.info(f"Writing memory log to {path}...")
        df.to_csv(path, index=False)



class CUDAMemoryTracker:

    def __init__(
        self,
        device: torch.device,
        output_dir: Path,
    ):
        self.device = device
        self.output_dir = output_dir

        self.logger = logging.getLogger(self.__class__.__name__)

        self.log: list[tuple[str, int, int]] = []

        if self.device.type == "cuda":
            self.track("initial")
        else:
            self.logger.warning("CUDAMemoryTracker initialized on non-CUDA device. Memory tracking will be disabled.")

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

        self.log.append((tag, mem, max_mem))

        self.logger.info(
            f"{tag}: {humanize.naturalsize(mem, binary=True)} "
            f"(Peak: {humanize.naturalsize(max_mem, binary=True)})"
        )

    def write(self, path: Path):
        df = pd.DataFrame(self.log, columns=["tag", "allocated", "max_allocated"])
        self.logger.info(f"Writing CUDA memory log to {path}...")
        df.to_csv(path, index=False)

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

        self.logger.info(log)

        self.logger.info(f"Writing CUDA memory summary to {path}...")
        with path.open('w') as file:
            file.write(log)

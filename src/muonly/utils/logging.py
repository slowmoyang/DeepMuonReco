import logging
import time
from getpass import getuser
from pathlib import Path
from socket import gethostname
import json
from contextlib import contextmanager
from aim.pytorch_lightning import AimLogger
from omegaconf import OmegaConf
from omegaconf import DictConfig


_logger = logging.getLogger(__name__)


@contextmanager
def elapsed_timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def log_everything(logger: AimLogger, config: DictConfig, model, output_dir: Path):
    logger.experiment.name = config.run.name

    logger.experiment.set(
        key="config",
        val=OmegaConf.to_container(config),  # type: ignore
    )
    logger.experiment.set(
        key="env",
        val={
            "host": gethostname(),
            "cwd": str(Path.cwd()),
            "user": getuser(),
        },
    )
    logger.experiment.set(
        key="model",
        val={
            "num_parameters": model.num_parameters,
        },
    )
    for tag in config.run.tags:
        logger.experiment.add_tag(tag)

    description_file = output_dir / "description.txt"
    if description_file.exists():
        _logger.info(f"Loading description from {description_file}")
        with open(description_file, "r") as stream:
            description = stream.read()
        _logger.info(f"{description=}")
        logger.experiment.description = description
    elif description := config.run.description:
        logger.experiment.description = description

# FIXME:
def is_json_serializable(obj) -> bool:
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

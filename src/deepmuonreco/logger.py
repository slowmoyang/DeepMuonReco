from logging import getLogger
from typing import Literal, Any
from pathlib import Path
import aim
from aim.sdk.run import warnings
from matplotlib.figure import Figure
import torch


Stage = Literal['train', 'val', 'test']

_logger = getLogger(__name__)


class Logger:

    def __init__(
        self,
        repo: str,
        experiment: str,
        run: str,
    ) -> None:
        """
        Args:
            repo (str): Aim repository path
            experiment (str): Aim experiment name
            run (str): Aim run name
        """
        self.run = aim.Run(
            repo=repo,
            experiment=experiment,
        )
        self.run.name = run

        self.log_dir = Path(repo, experiment, run)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.step = 0
        self.epoch = 0

    def track(
        self,
        stage: Stage,
        **kwargs: Any,
    ) -> None:

        loggable = {}
        for key in kwargs.keys():
            value = kwargs[key]
            if isinstance(value, Figure):
                loggable[key] = aim.Image(value)
            elif isinstance(value, tuple):
                if isinstance(value[0], Figure):
                    loggable[key] = aim.Image(value[0])
            elif torch.is_tensor(value):
                if value.numel() == 1:
                    loggable[key] = value.item()
                else:
                    _logger.warning(
                        f'Attempting to log a tensor with more than one element: {key}. '
                    )
            else:
                loggable[key] = value

        self.run.track(
            value=loggable,
            step=self.step,
            epoch=self.epoch,
            context={'subset': stage},
        )

        if stage == 'train':
            self.step += 1


    def __getitem__(self, key: str) -> Any:
        return self.run[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.run[key] = value

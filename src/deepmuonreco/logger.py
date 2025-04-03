from typing import Literal, Any
from pathlib import Path
import aim
from matplotlib.figure import Figure

Stage = Literal['train', 'val', 'test']

class Logger:

    def __init__(self) -> None:
        self.aim_run = aim.Run()

        self.log_dir = Path(
            self.aim_run.repo.path,
            self.aim_run.experiment,
            self.aim_run.hash,
        )
        self.log_dir.mkdir(parents=True)

        self.step = 0
        self.epoch = 0

    def track(
        self,
        stage: Stage,
        **kwargs: Any,
    ) -> None:
        for key in kwargs.keys():
            value = kwargs[key]
            if isinstance(value, Figure):
                kwargs[key] = aim.Image(value)
            elif isinstance(value, tuple):
                if isinstance(value[0], Figure):
                    kwargs[key] = aim.Image(value[0])

        self.aim_run.track(
            kwargs,
            step=self.step,
            epoch=self.epoch,
            context={'subset': stage},
        )

        if stage == 'train':
            self.step += 1


    def __getitem__(self, key: str) -> Any:
        return self.aim_run[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.aim_run[key] = value

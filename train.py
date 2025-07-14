#!/usr/bin/env python
from socket import gethostname
from getpass import getuser
import sys
from logging import getLogger
from typing import cast
from pathlib import Path
from aim.pytorch_lightning import AimLogger
import hydra
from hydra.utils import instantiate
import torch
from lightning.pytorch import LightningDataModule, Trainer
from lightning import seed_everything
from omegaconf import DictConfig
from omegaconf import OmegaConf
from coolname import generate_slug
from deepmuonreco.model import Model
from deepmuonreco.nn.utils import init_weights

_logger = getLogger(__name__)


OmegaConf.register_new_resolver(
    'slug',
    lambda pattern = 2: generate_slug(pattern=pattern),
    use_cache=True,
    replace=True,
)

OmegaConf.register_new_resolver(
    name='eval',
    resolver=eval,
)



def run(
    model: Model,
    datamodule: LightningDataModule,
    trainer: Trainer,
) -> None:
    trainer.validate(model=model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path='best')
    trainer.predict(model=model, datamodule=datamodule, ckpt_path='best')


@hydra.main(
    config_path='./config',
    config_name='ttbar-2024pu',
    version_base=None,
)
def main(config: DictConfig) -> None:
    _logger.info(' '.join(sys.argv))

    output_dir = Path(config.paths.output_dir)
    with open(output_dir / 'config.yaml', 'w') as stream:
        OmegaConf.save(config=config, f=stream)

    torch.set_num_threads(config.num_threads)

    seed_everything(config.seed, workers=True)

    model = Model.from_config(config=config)
    _logger.info(f'{model=}')
    _logger.info(f'{model.num_params=}')

    _logger.info('Initializing model weights')
    model.apply(init_weights)

    callback_dict = instantiate(config.callbacks)

    trainer: Trainer = instantiate(config.trainer)(
        callbacks=list(callback_dict.values()),
    )
    datamodule: LightningDataModule = instantiate(config.data.module)

    logger = cast(AimLogger, trainer.logger)
    logger.experiment.set(
        key='config',
        val=OmegaConf.to_container(config), # type: ignore
    )
    logger.experiment.set(
        key='env',
        val={
            'host': gethostname(),
            'cwd': str(Path.cwd()),
            'user': getuser(),
        },
    )
    logger.experiment.set(
        key='model',
        val={
            'num_params': model.num_params,
        }
    )
    for tag in config.tags:
        logger.experiment.add_tag(tag)

    description_file = output_dir / 'description.txt'
    if description_file.exists():
        with open(description_file, 'r') as stream:
            description = stream.read()
            _logger.info(f'{description=}')
            logger.experiment.description = description

    logger.experiment.description = config.description

    run(
        model=model,
        datamodule=datamodule,
        trainer=trainer,
    )



if __name__ == '__main__':
    main()

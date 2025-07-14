from logging import getLogger
from pathlib import Path
from typing import Any, Type, Callable
from functools import cached_property
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from .datasets import *

_logger = getLogger(__name__)


__all__ = [
    'InnerTrackSelectionDataModule',
]


class DataModule(LightningDataModule):

    dataset_cls: Type

    def __init__(
        self,
        root: str,
        batch_size: int,
        eval_batch_size: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_examples: int | None = None,
        val_examples: int | None = None,
        test_examples: int | None = None,
        predict_examples: int | None = None,
        train_sampler: Callable | None = None, # TODO:
        pre_fit_validation: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    @cached_property
    def root(self) -> Path:
        return Path(self.hparams['root'])

    @cached_property
    def eval_batch_size(self) -> int:
        return self.hparams['eval_batch_size'] or self.hparams['batch_size']

    @cached_property
    def train_set(self):
        path = self.root / "train.root"
        _logger.info(f'Loading training set from {path}')
        return self.dataset_cls(
            path=path,
            stop=self.hparams[f'train_examples']
        )

    @cached_property
    def val_set(self):
        path = self.root / "val.root"
        _logger.info(f'Loading validation set from {path}')
        return self.dataset_cls(
            path=path,
            stop=self.hparams[f'val_examples']
        )

    @cached_property
    def test_set(self):
        path = self.root / 'val.root'
        _logger.info(f'Loading test set from {path} to perform a full evaluation for model selection')
        return self.dataset_cls(
            path=path,
            stop=self.hparams[f'test_examples']
        )

    @cached_property
    def predict_set(self):
        path = self.root / 'test.root'
        _logger.info(f'Loading prediction set from {path} for the final evaluation')
        return self.dataset_cls(
            path=path,
            stop=self.hparams[f'predict_examples']
        )

    def setup(self, stage: str):
        if stage == 'fit':
            _logger.info('Setting up training and validation datasets')
            self.train_set
            self.val_set
        elif stage == 'validate':
            _logger.info('Setting up validation dataset')
            self.val_set
        elif stage == 'test':
            _logger.info('Setting up test dataset')
            self.test_set
        elif stage == 'predict':
            _logger.info('Setting up prediction dataset')
            self.predict_set
        else:
            raise RuntimeError(f'got an unexpected {stage=}')

    def teardown(self, stage: str):
        if stage == 'fit':
            delattr(self, 'train_set')
            delattr(self, 'val_set')
        elif stage == 'validate':
            if not self.hparams['pre_fit_validation']:
                _logger.info('Skipping validation set teardown as pre_fit_validation is False')
                delattr(self, 'val_set')
        elif stage == 'test':
            delattr(self, 'test_set')
        elif stage == 'predict':
            delattr(self, 'predict_set')
        else:
            raise RuntimeError(f'got an unexpected {stage=}')

    def train_dataloader(self):
        _logger.info('Creating training dataloader')
        dataset = self.train_set

        kwargs: dict[str, Any] = {}
        if train_sampler := self.hparams.get('train_sampler', None):
            _logger.info(f'Using custom train sampler: {train_sampler}')
            kwargs['sampler'] = train_sampler(dataset)
        else:
            kwargs['shuffle'] = True

        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams['batch_size'],
            drop_last=True,
            collate_fn=dataset.collate,
            pin_memory=self.hparams['pin_memory'],
            num_workers=self.hparams['num_workers'],
            **kwargs,
        )

    def _eval_dataloader(self, dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.eval_batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=dataset.collate,
            pin_memory=self.hparams['pin_memory'],
            num_workers=self.hparams['num_workers'],
        )

    def val_dataloader(self):
        _logger.info('Creating validation dataloader')
        return self._eval_dataloader(self.val_set)

    def test_dataloader(self):
        _logger.info('Creating test dataloader')
        return self._eval_dataloader(self.test_set)

    def predict_dataloader(self):
        _logger.info('Creating prediction dataloader')
        return self._eval_dataloader(self.predict_set)


class InnerTrackSelectionDataModule(DataModule):
    dataset_cls = InnerTrackSelectionDataset

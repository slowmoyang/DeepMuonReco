from logging import Logger
from typing import Any
import torch
from torch import Tensor
from torch.optim import Optimizer
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn import TensorDictSequential
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer.states import RunningStage
from torchmetrics import MetricCollection
from omegaconf import DictConfig
from aim.pytorch_lightning import AimLogger
from aim.storage.object import CustomObject
from hydra.utils import instantiate
#
from .utils.config import build_tensordictmodule
from .utils.config import build_tensordictsequential
from .utils.config import build_metric_collection
from .utils.optim import group_parameters


_logger: Logger = Logger(__name__)


class Model(LightningModule):

    def __init__(
        self,
        augmentation: TensorDictSequential,
        preprocessing: TensorDictSequential,
        postprocessing: TensorDictSequential,
        pre_metric_postprocessing: TensorDictSequential,
        model: TensorDictModule,
        criterion: TensorDictModule,
        criterion_reduction: TensorDictModule,
        train_metrics: MetricCollection,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        optimizer_config: DictConfig,
        lr_scheduler_config: DictConfig,
    ) -> None:
        super().__init__()

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.model = model
        self.criterion = criterion
        self.criterion_reduction = criterion_reduction
        self.pre_metric_postprocessing = pre_metric_postprocessing
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self._optimizer_config = optimizer_config
        self._lr_scheduler_config = lr_scheduler_config

        self.lr = optimizer_config.lr

    @classmethod
    def from_config(cls, config: DictConfig):
        augmentation = build_tensordictsequential(config.augmentation)
        preprocessing = build_tensordictsequential(config.preprocessing)
        postprocessing = build_tensordictsequential(config.postprocessing)
        model = build_tensordictmodule(config.model)
        criterion = build_tensordictmodule(config.criterion)
        criterion_reduction = build_tensordictmodule(config.criterion_reduction)
        pre_metric_postprocessing = build_tensordictsequential(config.pre_metric_postprocessing)
        train_metrics = build_metric_collection(config=config.metric.train, stage=RunningStage.TRAINING)
        val_metrics = build_metric_collection(config=config.metric.val, stage=RunningStage.VALIDATING)
        test_metrics = build_metric_collection(config=config.metric.test, stage=RunningStage.TESTING)
        optimizer_config = instantiate(config.optimizer)
        lr_scheduler_config = instantiate(config.lr_scheduler)

        return cls(
            augmentation=augmentation,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            model=model,
            criterion=criterion,
            criterion_reduction=criterion_reduction,
            pre_metric_postprocessing=pre_metric_postprocessing,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
        )

    def forward(self, batch: TensorDict) -> TensorDict:
        """
        """
        with torch.no_grad():
            batch = self.preprocessing(batch)
        batch = self.model(batch)
        return batch

    def training_step(self, batch: TensorDict) -> Tensor:
        with torch.no_grad():
            batch = self.augmentation(batch)
        batch = self(batch)
        batch = self.criterion(batch)
        loss = self.criterion_reduction(batch)['loss']
        self.train_metrics.update(batch)
        self.log(name=f'{RunningStage.TRAINING.value}_loss', value=loss)
        return loss

    def _eval_step(self, batch: TensorDict, metrics: MetricCollection) -> TensorDict:
        batch = self(batch)
        batch = self.criterion(batch)
        batch = self.postprocessing(batch)
        batch = self.pre_metric_postprocessing(batch)
        metrics.update(batch)
        return batch

    def validation_step(self, batch: TensorDict) -> TensorDict:
        return self._eval_step(batch=batch, metrics=self.val_metrics)

    def test_step(self, batch: TensorDict) -> TensorDict:
        return self._eval_step(batch=batch, metrics=self.test_metrics)

    def predict_step(self, batch: TensorDict) -> TensorDict:
        batch = self(batch)
        batch = self.postprocessing(batch)
        return batch

    def track(self, name: str, value: Any) -> None:
        if not isinstance(self.logger, AimLogger):
            raise ValueError('logger is not AimLogger')
        name, context = self.logger.parse_context(name)
        self.logger.experiment.track(
            value=value,
            name=name,
            step=self.global_step,
            epoch=self.current_epoch,
            context=context
        )

    def _log(self, key: str, value: Any, log_dict: dict) -> None:
        # FIXME: just warn, not raise error
        assert isinstance(self.logger, AimLogger), 'logger is not AimLogger'

        if isinstance(value, CustomObject):
            self.logger.experiment.track(value=value, name=key)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self._log(key=f'{key}_{sub_key}', value=sub_value, log_dict=log_dict)
        else:
            log_dict[key] = value

    def _on_epoch_end(self, metrics: MetricCollection) -> None:
        log_dict = {}
        for key, value in metrics.compute().items():
            self._log(key=key, value=value, log_dict=log_dict)
        if len(log_dict) > 0:
            self.log_dict(log_dict)
        metrics.reset()

    def on_train_epoch_start(self) -> None:
        self.track(value=self.current_epoch, name='epoch')

    def on_train_epoch_end(self) -> None:
        return self._on_epoch_end(self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        return self._on_epoch_end(self.val_metrics)

    def on_test_epoch_end(self) -> None:
        return self._on_epoch_end(self.test_metrics)

    def _configure_optimizer(self) -> Optimizer:
        config = self._optimizer_config
        params = group_parameters(
            model=self.model,
            weight_decay=config.weight_decay,
        )

        optimizer = config.optimizer(
            params=params,
            lr=self.lr,
        )
        return optimizer

    def _configure_lr_scheduler(self, optimizer: Optimizer):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        config = self._lr_scheduler_config
        if config is None:
            return None

        lr_scheduler_config = {
            'scheduler': config.scheduler(optimizer=optimizer),
        }

        for key in ['interval', 'frequency', 'monitor', 'strict', 'name']:
            if value := config.get(key, None):
                lr_scheduler_config[key] = value

        return lr_scheduler_config


    def configure_optimizers(self): # type: ignore[override]
        optimizer = self._configure_optimizer()

        _logger.info(f'{optimizer=}')

        if lr_scheduler_config := self._configure_lr_scheduler(optimizer=optimizer):
            _logger.info(f'{lr_scheduler_config=}')
            return {
                'optimizer': optimizer,
                'lr_scheduler': lr_scheduler_config,
            }
        else:
            return optimizer

    @property
    def num_params(self) -> int:
        return sum(each.numel() for each in self.model.parameters() if each.requires_grad)

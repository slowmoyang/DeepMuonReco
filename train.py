#!/usr/bin/env python
import random
from typing import Any
from logging import getLogger
import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.amp.grad_scaler import GradScaler
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchmetrics import MetricCollection
import tqdm
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from omegaconf import DictConfig
from coolname import generate_slug
from deepmuonreco.nn.utils import init_params
from deepmuonreco.logger import Logger, Stage
from deepmuonreco.utils.config import (
    build_tensordictmodule,
    build_tensordictsequential,
    build_metric_collection,
)


_logger = getLogger(__name__)


OmegaConf.register_new_resolver(
    name='slug',
    resolver=lambda pattern = 2: generate_slug(pattern=pattern),
    use_cache=True,
    replace=True,
)

OmegaConf.register_new_resolver(
    name='eval',
    resolver=eval,
    replace=True,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def train(
    preprocessing: TensorDictSequential,
    model: TensorDictModule,
    data_loader: DataLoader,
    criterion: TensorDictModule,
    criterion_reduction: TensorDictModule,
    grad_scaler: GradScaler,
    optimizer: Optimizer,
    device: torch.device,
    logger: Logger,
    config: DictConfig,
) -> None:
    """
    train the model for one epoch
    """
    model.train()

    # FIXME:
    autocast = torch.autocast(device_type='cuda', dtype=torch.float16)

    for idx, batch in enumerate(data_loader):
        batch = batch.to(device)

        with autocast:
            batch = preprocessing(batch)
            batch = model(batch)
            batch = criterion(batch)
            loss = criterion_reduction(batch)['loss']

        grad_scaler.scale(loss).backward()

        if config.fit.max_grad_norm != 0:
            grad_scaler.unscale_(optimizer)
            clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=config.fit.max_grad_norm,
            )

        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # log metrics
        if (idx + 1) % 100 == 0:
            logger.track(stage='train', loss=loss)


@torch.inference_mode()
def evaluate(
    model: TensorDictModule,
    preprocessing: TensorDictSequential,
    postprocessing: TensorDictSequential,
    data_loader: DataLoader,
    criterion: TensorDictModule,
    pre_metric_postprocessing: TensorDictSequential,
    metric_collection: MetricCollection,
    device: torch.device,
    logger: Logger,
    stage: Stage,
) -> dict[str, Any]:
    """
    """
    model.eval()

    for batch in data_loader:
        batch = batch.to(device)

        batch = preprocessing(batch)
        batch = model(batch)
        batch = criterion(batch)
        batch = postprocessing(batch)
        batch = pre_metric_postprocessing(batch)
        metric_collection.update(batch)

    log_dict = metric_collection.compute()
    metric_collection.reset()
    logger.track(stage=stage, **log_dict)
    return log_dict


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@hydra.main(
    config_path='./config',
    config_name='config',
    version_base=None,
)
def main(config: DictConfig) -> None:
    # logger
    logger = instantiate(config.logger)
    log_dir = logger.log_dir

    set_seed(config.seed)
    torch.set_num_threads(config.num_threads)

    with open(log_dir / 'config.yaml', 'w') as stream:
        OmegaConf.save(config=config, f=stream)

    # model
    model = build_tensordictmodule(config.model)
    _logger.info(f'{model=}')

    model.apply(init_params)
    logger['num_params'] = sum(each.numel() for each in model.parameters())

    # preprocessing, postprocessing
    preprocessing = build_tensordictsequential(config.preprocessing)
    postprocessing = build_tensordictsequential(config.postprocessing)

    # criterion & optimizer
    criterion = build_tensordictmodule(config.criterion)
    criterion_reduction = build_tensordictmodule(config.criterion_reduction)

    optimizer = instantiate(config.optimizer)(model.parameters())

    grad_scaler = GradScaler(device='cuda')

    # pre-metric postprocessing and metric collection for evaluation stages
    val_pre_metric_postprocessing = build_tensordictsequential(config.metric.val.postprocessing)
    val_metric_collection = build_metric_collection(config.metric.val.metric)

    # device
    device = torch.device(config.device)
    _logger.info(f'Using device: {device}')

    model = model.to(device)
    preprocessing = preprocessing.to(device)
    postprocessing = postprocessing.to(device)
    criterion = criterion.to(device)
    criterion_reduction = criterion_reduction.to(device)

    val_pre_metric_postprocessing = val_pre_metric_postprocessing.to(device)
    val_metric_collection = val_metric_collection.to(device)

    # datasets & data loaders
    val_set = instantiate(config.data.val_set)
    _logger.info(f'Validation set: {len(val_set)} examples')
    val_loader = instantiate(config.data.val_loader)(
        dataset=val_set,
        collate_fn=val_set.collate,
    )
    _logger.info(f'Validation loader: {len(val_loader)} batches')

    train_set = instantiate(config.data.train_set)
    _logger.info(f'Training dataset: {len(train_set)} examples')

    train_loader = instantiate(config.data.train_loader)(
        dataset=train_set,
        collate_fn=train_set.collate,
    )
    _logger.info(f'Training data loader: {len(train_loader)} batches')

    # checkpoint
    ckpt_dir = logger.log_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / 'best.pt'

    ############################################################################
    # training loop
    ############################################################################

    best_val_loss = float('inf')
    best_epoch = -1
    early_stopping_wait_count = 0

    for epoch in (pbar := tqdm.trange(0, config.fit.num_epochs + 1, desc='Epoch')):
        logger.epoch = epoch
        _logger.info(f'Starting epoch {epoch}...')

        if epoch >= 1:
            train(
                preprocessing=preprocessing,
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                criterion_reduction=criterion_reduction,
                grad_scaler=grad_scaler,
                optimizer=optimizer,
                device=device,
                logger=logger,
                config=config,
            )

        val_result = evaluate(
            preprocessing=preprocessing,
            model=model,
            postprocessing=postprocessing,
            data_loader=val_loader,
            criterion=criterion,
            pre_metric_postprocessing=val_pre_metric_postprocessing,
            metric_collection=val_metric_collection,
            device=device,
            logger=logger,
            stage='val',
        )

        # check for improvement
        if val_result['loss'] < best_val_loss:
            best_val_loss = val_result['loss']
            best_epoch = epoch

            pbar.set_description(
                f'epoch={epoch}: loss={val_result["loss"]:.4f} '
                f'(best_val_loss={best_val_loss:.4f} @ epoch={best_epoch})'
            )

            # save checkpoint
            torch.save(
                obj=dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    epoch=logger.epoch,
                    step=logger.step,
                    val_result=val_result,
                ),
                f=ckpt_path,
            )

            early_stopping_wait_count = 0
            _logger.debug(
                f'[EarlyStopping] Reset early stopping wait count to 0 at {epoch=} '
            )
        else:
            early_stopping_wait_count += 1
            # log earlying stoping status with count
            _logger.debug(
                f'[EarlyStopping] {early_stopping_wait_count=} '
                f'with {best_val_loss=:.4f} at {best_epoch=}'
            )
            if early_stopping_wait_count >= config.fit.early_stopping_patience:
                _logger.info(
                    f'[EarlyStopping] Early stopping at {epoch=} with {best_val_loss=:.4f}'
                    f' at {best_epoch=}.'
                )
                break

    ############################################################################
    # testing
    ############################################################################
    del (
        train_set, train_loader, val_set, val_loader,
        val_pre_metric_postprocessing, val_metric_collection,
    )

    test_set = instantiate(config.data.test_set)
    test_loader = instantiate(config.data.test_loader)(dataset=test_set, collate_fn=test_set.collate)

    test_pre_metric_postprocessing = build_tensordictsequential(config.metric.test.postprocessing)
    test_metric_collection = build_metric_collection(config.metric.test.metric)

    test_pre_metric_postprocessing = test_pre_metric_postprocessing.to(device)
    test_metric_collection = test_metric_collection.to(device)

    best_checkpoint = torch.load(ckpt_path)
    model.load_state_dict(
        state_dict=best_checkpoint['model'],
    )
    del best_checkpoint

    test_result = evaluate(
        preprocessing=preprocessing,
        model=model,
        postprocessing=postprocessing,
        data_loader=test_loader,
        criterion=criterion,
        pre_metric_postprocessing=test_pre_metric_postprocessing,
        metric_collection=test_metric_collection,
        device=device,
        logger=logger,
        stage='test',
    )


    torch.save(
        obj=test_result,
        f=log_dir / 'test_result.pt',
    )

    _logger.info(f'{test_result=}')


if __name__ == '__main__':
    main()

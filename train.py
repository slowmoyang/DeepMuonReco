#!/usr/bin/env python
"""
inner track selection task

plan to implement a custom trainer or switch to pytorch_lightning after people
get familiar with deep learning codes
"""
import random
from typing import Any
import numpy as np
import torch
import torch.nn
from torch.nn import Sigmoid
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchmetrics import MetricCollection
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryROC
import tqdm
#
from deepmuonreco.data import InnerTrackSelectionDataset
from deepmuonreco.nn import Normalize
from deepmuonreco.nn import InnerTrackSelectionTransformer
from deepmuonreco.nn.utils import init_params
from deepmuonreco.nn import SelectedBCEWithLogitsLoss
from deepmuonreco.logger import Logger


def train(
    model: TensorDictSequential,
    data_loader: DataLoader,
    criterion: TensorDictModule,
    optimizer: Optimizer,
    device: torch.device,
    logger: Logger,
) -> None:
    model.train()

    for idx, batch in enumerate(data_loader):
        batch = batch.to(device)

        # model inference
        batch = model(batch)

        # compute loss
        loss_batch = criterion(batch)
        loss = loss_batch['loss'].mean()

        # backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # log metrics
        # FIXME:
        if (idx + 1) % 1 == 0:
            logger.track(stage='train', loss=loss)


@torch.no_grad()
def validate(
    model: TensorDictSequential,
    data_loader: DataLoader,
    criterion: TensorDictModule,
    loss_metric: MeanMetric,
    roc_metric: MetricCollection,
    device: torch.device,
    logger: Logger,
) -> dict[str, Any]:
    """
    """
    model.eval()

    for batch in data_loader:
        batch = batch.to(device)

        # model inference
        batch = model(batch)

        # compute loss
        loss_batch = criterion(batch)

        # compute metrics
        mask = batch['masks']['track']
        preds = batch['score'].masked_select(mask)
        target = batch['target'].long().masked_select(mask)

        loss_metric.update(loss_batch['loss'])
        roc_metric.update(
            preds=preds,
            target=target,
        )

    # log metrics
    log_dict = {}
    log_dict['loss'] = loss_metric.compute().cpu()
    log_dict['roc_auc'] = roc_metric['auc'].compute().cpu()
    logger.track(stage='val', **log_dict)

    loss_metric.reset()
    roc_metric.reset()

    return log_dict

@torch.no_grad()
def test(
    model: TensorDictSequential,
    data_loader: DataLoader,
    criterion: TensorDictModule,
    loss_metric: MeanMetric,
    roc_metric: MetricCollection,
    device: torch.device,
    logger: Logger,
) -> dict[str, Any]:
    """
    """
    model.eval()

    for batch in data_loader:
        batch = batch.to(device)

        batch = model(batch)
        loss_batch = criterion(batch)

        mask = batch['masks']['track']
        preds = batch['score'].masked_select(mask)
        target = batch['target'].long().masked_select(mask)

        loss_metric.update(loss_batch['loss'])
        roc_metric.update(
            preds=preds,
            target=target,
        )

    log_dict: dict[str, Any] = {}
    log_dict['loss'] = loss_metric.compute().cpu()
    log_dict['roc_auc'] = roc_metric['auc'].compute().cpu()
    log_dict['roc_curve'] = roc_metric['curve'].plot()
    logger.track(stage='test', **log_dict)

    loss_metric.reset()
    roc_metric.reset()

    return log_dict

###############################################################################
# hyperparameters
###############################################################################
num_epochs = 100
device = torch.device('cuda:0')
train_file_path = './data/sanity-check.root'
val_file_path = './data/sanity-check.root'
batch_size = 32
eval_batch_size = 32
early_stopping_patience = 10
seed = 1337

dim_model = 64
dim_feedforward = 128
activation = 'gelu'
num_heads = 4
num_layers = 2
dropout = 0.1

# NOTE: see https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
# TODO: pos_weight must be calculated from dataset with enough statistics
pos_weight = torch.tensor(30)

###############################################################################
# for reproducibility
###############################################################################
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

###############################################################################
# aim-based logger
###############################################################################
logger = Logger()

###############################################################################
# datasets & data loaders
###############################################################################
train_set = InnerTrackSelectionDataset(path=train_file_path)
val_set = InnerTrackSelectionDataset(path=val_file_path)

collate_fn = InnerTrackSelectionDataset.collate

train_loader = DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    dataset=val_set,
    batch_size=eval_batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
)

###############################################################################
# preprocessing -> model -> postprocessing
###############################################################################
preprocessor = TensorDictSequential([
    TensorDictModule(
        module=Normalize(
            mean=[0, 0, 0], # [px, py, pz]
            std=[100, 100, 3],
        ),
        in_keys=['track'],
        out_keys=['track'],
    ),
    TensorDictModule(
        module=Normalize(
            mean=[0, 0, 0, 0, 0, 0], # [pox_x, pos_py, pos_z, dir_x, dir_y, dir_z]
            std=[750, 720, 1000, 1, 1, 1],
        ),
        in_keys=['segment'],
        out_keys=['segment'],
    ),
])

# actual model
raw_model = TensorDictModule(
    module=InnerTrackSelectionTransformer(
        dim_model=dim_model,
        dim_feedforward=dim_feedforward,
        activation=activation,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ),
    in_keys=[
        'track',
        ('pad_masks', 'track'),
        'segment',
        ('pad_masks', 'segment'),
    ],
    out_keys=[
        'logits'
    ],
)

raw_model.apply(init_params)

# postprocessing
postprocessor = TensorDictSequential([
    TensorDictModule(
        module=Sigmoid(),
        in_keys=['logits'],
        out_keys=['score'],
    ),
])


# model containing preprocessing and postprocessing
model = TensorDictSequential([
    preprocessor,
    raw_model,
    postprocessor,
])


################################################################################
# loss function & optimizer
################################################################################
criterion = TensorDictModule(
    module=SelectedBCEWithLogitsLoss(pos_weight=pos_weight),
    in_keys=[
        'logits',
        'target',
        ('masks', 'track'),
    ],
    out_keys=[
        'loss',
    ],
    inplace=False,
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-1,
)

################################################################################
# metrics
################################################################################
loss_metric = MeanMetric()

roc_metric = MetricCollection(
    metrics={
        'curve': BinaryROC(compute_on_cpu=True),
        'auc': BinaryAUROC(compute_on_cpu=True),
    },
    compute_groups=True,
)

################################################################################
# move to gpu
################################################################################
model = model.to(device)
criterion = criterion.to(device)
preprocessor = preprocessor.to(device)
postprocessor = postprocessor.to(device)
loss_metrics = loss_metric.to(device)
roc_metrics = roc_metric.to(device)

################################################################################
# summary
################################################################################
logger['num_params'] = sum(each.numel() for each in model.parameters())

################################################################################
# training
################################################################################
best_val_loss = float('inf')
best_epoch = -1

early_stopping_wait_count = 0
checkpoint_path = logger.log_dir / 'best_model.pt'

for epoch in (pbar := tqdm.trange(0, num_epochs + 1, desc='Epoch')):
    logger.epoch = epoch

    if epoch >= 1:
        train(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logger=logger,
        )

    val_result = validate(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        loss_metric=loss_metric,
        roc_metric=roc_metric,
        device=device,
        logger=logger,
    )

    # check if validation loss is improved
    if val_result['loss'] < best_val_loss:
        best_val_loss = val_result['loss']
        best_epoch = epoch

        pbar.set_description(
            f'epoch={epoch}: loss={val_result["loss"]:.4f}'
            f' ({best_val_loss=:.4f} @ epoch={best_epoch})'
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
            f=checkpoint_path,
        )

        early_stopping_wait_count = 0
    else:
        early_stopping_wait_count += 1
        if early_stopping_wait_count >= early_stopping_patience:
            break

################################################################################
# testing
################################################################################
best_checkpoint = torch.load(checkpoint_path)
model.load_state_dict(
    state_dict=best_checkpoint['model'],
)
del best_checkpoint

test_result = test(
    model=model,
    data_loader=val_loader,
    criterion=criterion,
    loss_metric=loss_metric,
    roc_metric=roc_metric,
    device=device,
    logger=logger,
)

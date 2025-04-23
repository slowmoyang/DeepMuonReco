#!/usr/bin/env python
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
import argparse
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

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # log metrics
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
def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepMuonReco model with specified hyperparameters.")
    # Training loop parameters
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for training (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument("--train_file_path", type=str, default="./data/sanity-check.root", help="Path to training file")
    parser.add_argument("--val_file_path", type=str, default="./data/sanity-check.root", help="Path to validation file")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")

    # Model architecture parameters
    parser.add_argument("--dim_model", type=int, default=64, help="Dimension of the model embeddings")
    parser.add_argument("--dim_feedforward", type=int, default=128, help="Dimension of the feed-forward network")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function (e.g., 'gelu', 'relu')")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads for multi-head attention")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")

    # Optimization parameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay coefficient")
    parser.add_argument("--pos_weight", type=float, default=30, help="Positive weight for BCE loss")
    return parser.parse_args()

args = parse_args()

# NOTE: see https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
# TODO: pos_weight must be calculated from dataset with enough statistics
pos_weight = torch.tensor(args.pos_weight)

###############################################################################
# for reproducibility
###############################################################################
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

###############################################################################
# aim-based logger
###############################################################################
logger = Logger()

###############################################################################
# datasets & data loaders
###############################################################################
train_set = InnerTrackSelectionDataset(path=args.train_file_path)
val_set = InnerTrackSelectionDataset(path=args.val_file_path)

collate_fn = InnerTrackSelectionDataset.collate

train_loader = DataLoader(
    dataset=train_set,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    dataset=val_set,
    batch_size=args.eval_batch_size,
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
            mean=[0, 0, 0],  # for track: [px, py, eta]
            std=[100, 100, 3],
        ),
        in_keys=['track'],
        out_keys=['track'],
    ),
    TensorDictModule(
        module=Normalize(
            mean=[0, 0, 0, 0, 0, 0],  # for segment: [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z]
            std=[750, 720, 1000, 1, 1, 1],
        ),
        in_keys=['segment'],
        out_keys=['segment'],
    ),
    TensorDictModule(
        module=Normalize(
            mean=[0, 0, 0],  # for rechit: [pos_x, pos_y, pos_z]
            std=[750, 720, 1000],
        ),
        in_keys=['rechit'],
        out_keys=['rechit'],
    ),
])

# actual model
raw_model = TensorDictModule(
    module=InnerTrackSelectionTransformer(
        dim_model=args.dim_model,
        dim_feedforward=args.dim_feedforward,
        activation=args.activation,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ),
    in_keys=[
        'track',
        ('pad_masks', 'track'),
        'segment',
        ('pad_masks', 'segment'),
        'rechit',
        ('pad_masks', 'rechit'),
    ],
    out_keys=[
        'logits'
    ],
)

raw_model.apply(init_params)

# postprocessing: apply Sigmoid activation for score
postprocessor = TensorDictSequential([
    TensorDictModule(
        module=Sigmoid(),
        in_keys=['logits'],
        out_keys=['score'],
    ),
])

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
    lr=args.lr,
    weight_decay=args.weight_decay,
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
# logger
################################################################################
checkpoint_path = logger.log_dir / 'best_model.pt'
logger['checkpoint_path'] = str(checkpoint_path)
logger['hyperparameters'] = vars(args)
logger['num_params'] = sum(each.numel() for each in model.parameters())

################################################################################
# move to device
################################################################################
model = model.to(args.device)
criterion = criterion.to(args.device)
preprocessor = preprocessor.to(args.device)
postprocessor = postprocessor.to(args.device)
loss_metric = loss_metric.to(args.device)
roc_metric = roc_metric.to(args.device)

################################################################################
# training loop
################################################################################
best_val_loss = float('inf')
best_epoch = -1
early_stopping_wait_count = 0

for epoch in (pbar := tqdm.trange(0, args.num_epochs + 1, desc='Epoch')):
    logger.epoch = epoch

    if epoch >= 1:
        train(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=args.device,
            logger=logger,
        )

    val_result = validate(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        loss_metric=loss_metric,
        roc_metric=roc_metric,
        device=args.device,
        logger=logger,
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
            f=checkpoint_path,
        )

        early_stopping_wait_count = 0
    else:
        early_stopping_wait_count += 1
        if early_stopping_wait_count >= args.early_stopping_patience:
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
    device=args.device,
    logger=logger,
)

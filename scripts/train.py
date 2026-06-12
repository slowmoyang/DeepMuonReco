import logging
from pathlib import Path
from dataclasses import asdict, dataclass
import warnings
import os
from typing import Any
from contextlib import nullcontext
import secrets
import sys

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR
from torch.utils.data import DataLoader

from tensordict.nn import TensorDictModule

from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAUROC
from torchmetrics.classification import BinarySpecificityAtSensitivity

import torchinfo

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from coolname import generate_slug

import aim
from aim.sdk.objects.image import Image

import tqdm.rich
from tqdm import TqdmExperimentalWarning

from hist import Hist

import matplotlib.pyplot as plt
import mplhep as mh

from muonly.data.datasets import TrackerTrackSelectionDataset
from muonly.data.utils import configure_model_in_keys
from muonly.data.transforms.utils import configure_preprocessing
from muonly.utils.optim import configure_optimizer
from muonly.utils.optim import configure_lr_scheduler
from muonly.callbacks import ModelCheckpoint
from muonly.callbacks import MemoryTracker
from muonly.callbacks import CUDAMemoryTracker
from muonly.callbacks import TrackerCollection
from muonly.utils.reproducibility import set_seed

mh.style.use("CMS")

#===============================================================================
#
#===============================================================================

# FIXME: better way to set project root for hydra
os.environ["PROJECT_ROOT"] = str(Path(__file__).parents[1].resolve())

#===============================================================================
# logging
#===============================================================================
_logger = logging.getLogger(Path(__file__).name)
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

for name in ["matplotlib", "PIL", "aim", "h5py", "filelock"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

#===============================================================================
# resolvers
#===============================================================================

OmegaConf.register_new_resolver(
    name="slug",
    resolver=lambda pattern=2: generate_slug(pattern=pattern),
    use_cache=True,
    replace=True,
)

OmegaConf.register_new_resolver(
    name="randbits",
    resolver=lambda k=32: secrets.randbits(k=k),
    use_cache=True,
    replace=True,
)

OmegaConf.register_new_resolver(
    name="len",
    resolver=len,
)

#===============================================================================
#
#===============================================================================

@dataclass(repr=True)
class GlobalState:
    """Class to hold global state information such as current step and epoch."""

    step: int = 0
    epoch: int = 0

    def state_dict(self):
        return asdict(self)


#===============================================================================
#
#===============================================================================


def compute_loss(batch, criterion, config: DictConfig) -> Tensor:
    mask = batch["tracker_track_data_mask"]
    logits = batch["logits"]
    target = batch["target"].float()
    pt = batch["tracker_track_pt"]

    if config.loss.get("balancing") is not None:
        if config.loss.balancing.type == "pt_bin":
            pt_edges = config.loss.balancing.edges
            loss_list = []
            for pt_low, pt_high in zip(pt_edges[:-1], pt_edges[1:]):
                pt_low = pt_low or 0
                pt_high = pt_high or float("inf")

                bin_mask = mask & (pt > pt_low) & (pt <= pt_high)
                loss_list.append(
                    criterion(input=logits[bin_mask], target=target[bin_mask]).mean()
                )
            loss = torch.stack(loss_list).mean()
        else:
            raise ValueError(
                f"Unsupported loss balancing type: {config.loss.balancing.type}"
            )
    else:
        logits = batch["logits"][mask]
        target = batch["target"][mask].float()
        loss = criterion(input=logits, target=target)
        loss = loss.mean()
    return loss


#===============================================================================
#
#===============================================================================


def train(
    model,
    criterion: nn.Module,
    data_loader: DataLoader,
    optimizer: AdamW,
    lr_scheduler: SequentialLR,
    device: torch.device,
    global_state: GlobalState,
    config: DictConfig,
    amp_context: torch.autocast | nullcontext,
    aim_run: aim.Run,
) -> None:
    """ """
    model.train()

    progress_bar = tqdm.rich.tqdm(
        data_loader,
        desc="Training",
        disable=(not config.misc.tqdm),
    )

    for batch_idx, batch in enumerate(progress_bar):
        batch = batch.to(device)

        with amp_context:
            batch = model(batch)
            loss = compute_loss(batch=batch, criterion=criterion, config=config)

        loss.backward()
        clip_grad_norm_(
            parameters=model.parameters(), max_norm=config.optim.max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        global_state.step += 1

        aim_run.track(
            value=loss.float().item(),
            name="loss",
            epoch=global_state.epoch,
            step=global_state.step,
            context={"subset": "train"},
        )


#===============================================================================
#
#===============================================================================


@torch.inference_mode()
def validate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    config: DictConfig,
    amp_context: torch.autocast | nullcontext,
) -> dict[str, Any]:
    """ """
    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    model.eval()

    #---------------------------------------------------------------------------
    # Metrics and histogram setup
    #---------------------------------------------------------------------------
    metric_dict = {
        "loss": MeanMetric(),
        "loss_pt_0p5_3": MeanMetric(),
        "loss_pt_3_inf": MeanMetric(),
        "auroc_pt_3_inf": BinaryAUROC(thresholds=1_000),
    }

    h_sig = Hist.new.Reg(40, 0, 1).Double()
    h_bkg = h_sig.copy()

    # specificity = true negative rate = background rejection rate
    # sensitivity = true positive rate = signal efficiency
    sas_dict = {
        key: BinarySpecificityAtSensitivity(min_sensitivity=0.99_9, thresholds=None)
        for key in ['pt_0p5_3', 'pt_3_inf']
    }

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------

    for batch in tqdm.rich.tqdm(
        data_loader, desc="Evaluating", disable=(not config.misc.tqdm)
    ):
        batch = batch.to(device)

        with amp_context:
            batch = model(batch)

            mask = batch["tracker_track_data_mask"]
            logits = batch["logits"][mask]
            target = batch["target"][mask].float()
            preds = logits.sigmoid()
            loss = criterion(input=logits, target=target)

        #-----------------------------------------------------------------------
        #
        #-----------------------------------------------------------------------
        batch = batch.cpu()
        loss = loss.float().cpu()
        preds = preds.float().cpu()
        target = target.long().cpu()
        mask = mask.cpu()
        pt = batch["tracker_track_pt"][mask].float()

        mask_pt_0p5_3 = (pt > 0.5) & (pt <= 3)
        mask_pt_3_inf = pt > 3

        metric_dict["loss"].update(loss)
        metric_dict["loss_pt_0p5_3"].update(loss[mask_pt_0p5_3])
        metric_dict["loss_pt_3_inf"].update(loss[mask_pt_3_inf])
        metric_dict["auroc_pt_3_inf"].update(
            preds=preds[mask_pt_3_inf], target=target[mask_pt_3_inf]
        )

        sas_dict['pt_0p5_3'].update(preds=preds[mask_pt_0p5_3], target=target[mask_pt_0p5_3])
        sas_dict['pt_3_inf'].update(preds=preds[mask_pt_3_inf], target=target[mask_pt_3_inf])

        # numpy
        sig_mask = target == 1
        bkg_mask = torch.logical_not(sig_mask)

        h_sig.fill(preds[sig_mask].numpy())
        h_bkg.fill(preds[bkg_mask].numpy())

    #---------------------------------------------------------------------------
    #
    #---------------------------------------------------------------------------
    result = {name: metric.compute().item() for name, metric in metric_dict.items()}
    for key, value in sas_dict.items():
        tnr, threshold = value.compute()
        result[f"tnr_at_tpr_99p9_{key}"] = tnr.item()

    # NOTE:
    fig, ax = plt.subplots()
    hist_plot_kwargs: dict[str, Any] = dict(ax=ax, histtype="step", density=True)
    h_bkg.plot(label="Background Tracks", color="tab:blue", **hist_plot_kwargs)
    h_sig.plot(label="Signal Tracks", color="tab:orange", **hist_plot_kwargs)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend()
    result["dist_score"] = fig

    return result

#===============================================================================
#
#===============================================================================

def compute_pos_weight(dataset: TrackerTrackSelectionDataset, config) -> Tensor:
    pos_count = 0
    neg_count = 0

    for example in tqdm.rich.tqdm(dataset, desc="Computing pos_weight"):
        target = example["target"]

        if config.loss.get("balancing") is not None:
            if config.loss.balancing.type == "pt_bin":
                pt = example["tracker_track_pt"]
                # The loss only covers tracks within the pt bin range, so count
                # over the union of the bins defined by ``edges``.
                edges = config.loss.balancing.edges
                mask = (pt > edges[0]) & (pt <= edges[-1])
            else:
                raise ValueError(
                    f"Unsupported loss balancing type: {config.loss.balancing.type}"
                )

            target = target[mask]

        pos = target.long().sum().item()
        total = target.numel()
        neg = total - pos

        pos_count += pos
        neg_count += neg

    if pos_count == 0:
        raise ValueError(
            "No positive examples found in the dataset. Cannot compute pos_weight."
        )
    if neg_count == 0:
        raise ValueError(
            "No negative examples found in the dataset. Cannot compute pos_weight."
        )

    pos_weight = neg_count / pos_count

    return torch.tensor(pos_weight)


def run(
    aim_run: aim.Run,
    config: DictConfig,
) -> None:
    """
    Args:
        aim_run: Aim run object for logging and tracking.
        config: Configuration object containing all settings for the run.
    Returns:
        None
    """
    #---------------------------------------------------------------------------
    # Log
    #---------------------------------------------------------------------------
    _logger.info(f"{config=}")
    _logger.info(f"{sys.argv=}")

    #---------------------------------------------------------------------------
    # Run directory
    #---------------------------------------------------------------------------
    run_dir = Path(config.paths.run_dir)

    #---------------------------------------------------------------------------
    # Device setup
    #---------------------------------------------------------------------------
    device = torch.device(config.torch.device)

    _logger.info(f"{device=}")

    if device.type == "cuda":
        _logger.info(f"GPU Name: {torch.cuda.get_device_name(device)}")
    elif device.type == "cpu":
        _logger.warning(
            "Using CPU for training. This may be very slow. Consider using a GPU if possible."
        )
    else:
        _logger.warning(
            f"Using device of type {device.type}. Make sure this is intentional and that the device is properly configured."
        )

    #---------------------------------------------------------------------------
    # Setup resource trackers
    #---------------------------------------------------------------------------
    trackers = TrackerCollection()
    trackers += MemoryTracker(output_dir=run_dir)
    if device.type == "cuda":
        trackers += CUDAMemoryTracker(device=device, output_dir=run_dir)

    # warmup gpus
    if device.type == "cuda":
        _logger.info("Warming up GPU...")
        torch.empty(0, device=device)
        trackers.track("gpu_warmup")
        _logger.info("GPU warmup completed.")

    #---------------------------------------------------------------------------
    # Save config
    #---------------------------------------------------------------------------
    with open(run_dir / "config.yaml", "w") as file:
        OmegaConf.save(config=config, f=file, resolve=True)

    aim_run.name = config.run
    aim_run["config"] = OmegaConf.to_container(config, resolve=True)

    #---------------------------------------------------------------------------
    # PyTorch setup
    #---------------------------------------------------------------------------
    torch.set_float32_matmul_precision(config.torch.float32_matmul_precision)

    if config.torch.num_threads is not None:
        torch.set_num_threads(config.torch.num_threads)

    if config.torch.num_interop_threads is not None:
        torch.set_num_interop_threads(config.torch.num_interop_threads)

    set_seed(config.torch.seed)

    #---------------------------------------------------------------------------
    # Instantiate model
    #---------------------------------------------------------------------------
    model = instantiate(config.model)
    trackers.track("model_instantiated")

    # summarize model
    model_statistics = torchinfo.summary(model=model, verbose=False)
    _logger.info(f"Model summary:\n{model_statistics}")

    with open(run_dir / "model-summary.txt", "w") as file:
        file.write(f"{model_statistics}\n\nModel architecture:\n{model}")

    #---------------------------------------------------------------------------
    # Model compilation
    # FIXME: failed to compile the model with torch 2.10.0 on khu
    #---------------------------------------------------------------------------
    if config.torch.compile:
        _logger.warning(
            "Model compilation might cause many issues. Make sure to test the compiled model thoroughly before using it for training."
        )
        _logger.info("Compiling model with torch.compile...")
        compiled_model = torch.compile(model, mode="reduce-overhead")
        _logger.info("Model compilation completed.")
    else:
        compiled_model = None
        _logger.info("Model compilation is disabled. Using original model.")

    #---------------------------------------------------------------------------
    # Wrap model with TensorDictModule to handle input and output keys
    #---------------------------------------------------------------------------

    td_model = TensorDictModule(
        module=(compiled_model if compiled_model is not None else model),
        in_keys=configure_model_in_keys(config=config),
        out_keys=[
            "logits",
        ],
    )
    trackers.track("model_tensor_dict")

    # ---------------------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------------------
    train_set = TrackerTrackSelectionDataset(
        path=config.paths.train_file,
        config=config.data,
        max_events=config.data_load.train_max_events,
    )
    trackers.track("train_set_instantiated")
    _logger.info(f"Number of training examples: {len(train_set)}")
    train_set.summarize(path=(run_dir / "train-set-summary.json"), verbose=True)

    val_set = TrackerTrackSelectionDataset(
        path=config.paths.val_file,
        config=config.data,
        max_events=config.data_load.val_max_events,
    )
    trackers.track("val_set_instantiated")
    _logger.info(f"Number of validation examples: {len(val_set)}")

    #---------------------------------------------------------------------------
    # Preprocessing
    #---------------------------------------------------------------------------
    preprocessing = configure_preprocessing(config.data)

    train_set.apply_(preprocessing)
    trackers.track("preprocessing training set completed")

    val_set.apply_(preprocessing)
    trackers.track("preprocessing validation set completed")

    #---------------------------------------------------------------------------
    # Data loaders
    #---------------------------------------------------------------------------
    pin_memory = config.data_load.pin_memory and (device.type == "cuda")
    _logger.info(f"{pin_memory=}")

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.data_load.batch_size,
        shuffle=True,
        num_workers=config.data_load.num_workers,
        collate_fn=train_set.collate,
        pin_memory=pin_memory,
        drop_last=True,  # drop last to avoid issues stem from small non-representative batches during training
        generator=torch.Generator().manual_seed(config.torch.seed),
    )
    _logger.info(f"Number of training batches: {len(train_loader)}")

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=config.data_load.eval_batch_size,
        shuffle=False,  # no shuffling for evaluation
        num_workers=config.data_load.num_workers,
        collate_fn=val_set.collate,
        pin_memory=pin_memory,
        drop_last=False,  # we want to validate on all validation examples
    )
    _logger.info(f"Number of validation batches: {len(val_loader)}")

    #---------------------------------------------------------------------------
    # criterion
    #---------------------------------------------------------------------------
    if config.loss.pos_weight == "auto":
        pos_weight = compute_pos_weight(train_set, config)
    elif isinstance(config.loss.pos_weight, (int, float)):
        pos_weight = torch.tensor(config.loss.pos_weight)
    else:
        raise ValueError(f"Unsupported pos_weight value: {config.loss.pos_weight}")

    _logger.info(f"Positive weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
        reduction="none",
    )

    _logger.info(f"{criterion=}")

    #---------------------------------------------------------------------------
    # Optimization
    #---------------------------------------------------------------------------
    optimizer = configure_optimizer(
        model=model,
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        beta1=config.optim.beta1,
        beta2=config.optim.beta2,
    )
    _logger.info(f"{optimizer=}")

    #---------------------------------------------------------------------------
    # Learning rate scheduler
    #---------------------------------------------------------------------------
    lr_scheduler = configure_lr_scheduler(
        optimizer=optimizer,
        num_steps_per_epoch=len(train_loader),
        max_epochs=config.optim.max_epochs,
        max_lr=config.optim.lr,
        warmup_frac=config.optim.warmup.frac_steps,
        warmup_start_factor=config.optim.warmup.start_factor,
        annealing_eta_min_factor=config.optim.annealing.eta_min_factor,
    )

    #---------------------------------------------------------------------------
    # GPU setup
    #---------------------------------------------------------------------------
    _logger.debug("Moving model and criterion to device...")

    trackers.track("before_move_to_device")

    model = model.to(device)
    criterion = criterion.to(device)

    trackers.track("after_move_to_device")

    #---------------------------------------------------------------------------
    # Mixed precision setup
    #---------------------------------------------------------------------------
    if config.torch.precision == "float32":
        _logger.info("Using full precision (float32) for training.")
        amp_context = nullcontext()
    elif config.torch.precision == "bfloat16":
        _logger.info("Using mixed precision (bfloat16) for training.")
        amp_context = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported precision: {config.torch.precision}")

    #---------------------------------------------------------------------------
    # Global state setup
    #---------------------------------------------------------------------------
    global_state = GlobalState()

    #---------------------------------------------------------------------------
    # Checkpointing setup
    #---------------------------------------------------------------------------
    ckpt_dir = run_dir / "checkpoints"

    model_checkpoint = ModelCheckpoint(
        object_dict={
            "model": model,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "global_state": global_state,
        },
        metric="loss",
        mode="min",
        output_dir=ckpt_dir,
    )

    #---------------------------------------------------------------------------
    # Training loop
    #---------------------------------------------------------------------------

    trackers.track("training_loop_start")

    for epoch in tqdm.rich.trange(0, 1 + config.optim.max_epochs, desc="Epoch"):
        _logger.info(f"Starting epoch {epoch}/{config.optim.max_epochs}...")
        global_state.epoch = epoch

        if epoch >= 1:
            _logger.info("Running training...")
            train(
                model=td_model,
                criterion=criterion,
                data_loader=train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device=device,
                global_state=global_state,
                aim_run=aim_run,
                config=config,
                amp_context=amp_context,
            )
            _logger.info("Training completed.")
            trackers.track(f"epoch_{epoch:06d}_train")

        _logger.info("Running validation...")
        val_result = validate(
            model=td_model,
            criterion=criterion,
            data_loader=val_loader,
            device=device,
            config=config,
            amp_context=amp_context,
        )
        _logger.info("Validation completed.")
        trackers.track(f"epoch_{epoch:06d}_val")

        _logger.debug(
            f"Logging validation results to Aim and checkpointing if necessary..."
        )
        for key, value in val_result.items():
            if isinstance(value, plt.Figure):
                value = Image(value)
            aim_run.track(
                value=value,
                name=key,
                epoch=global_state.epoch,
                step=global_state.step,
                context={"subset": "val"},
            )
        plt.close("all")
        model_checkpoint.step(metric=val_result)
        _logger.info(f"{global_state}: {val_result}")

        _logger.debug("Logging completed")

        _logger.info(f"Epoch {epoch} completed.")

    trackers.track("training_loop_end")

    #---------------------------------------------------------------------------
    # Load best checkpoint
    #---------------------------------------------------------------------------

    checkpoint = torch.load(model_checkpoint.best_path)
    model.load_state_dict(checkpoint["model"])
    del checkpoint

    #---------------------------------------------------------------------------
    # Run final evaluation on validation set with best model checkpoint
    #---------------------------------------------------------------------------
    #
    # # TODO:
    # result = validate(
    #     model=td_model,
    #     criterion=criterion,
    #     data_loader=val_loader,
    #     device=device,
    #     config=config,
    #     amp_context=amp_context,
    # )
    #
    # for key, value in result.items():
    #     if isinstance(value, plt.Figure):
    #         value = Image(value)
    #     aim_run.track(
    #         value=value,
    #         name=key,
    #         epoch=global_state.epoch,
    #         step=global_state.step,
    #         context={"subset": "val"},
    #     )
    # plt.close("all")


@hydra.main(
    config_path="../config",
    config_name="no-hit",
    version_base=None,
)
def main(config: DictConfig) -> None:

    aim_run = aim.Run(
        repo=config.paths.log_dir,
        experiment=config.exp,
        log_system_params=True,
        capture_terminal_logs=True,
    )

    try:
        if config.torch.detect_anomaly:
            _logger.warning(
                "PyTorch anomaly detection is enabled. This may significantly "
                "slow down training. Use with caution."
            )

        with torch.autograd.set_detect_anomaly(
            mode=config.torch.detect_anomaly, check_nan=config.torch.check_nan
        ):
            run(aim_run=aim_run, config=config)
    except Exception as error:
        _logger.exception(f"An error occurred during training: {error}")
        aim_run.close()
        raise error
    finally:
        _logger.info("Closing Aim run.")
        aim_run.close()


if __name__ == "__main__":
    main()

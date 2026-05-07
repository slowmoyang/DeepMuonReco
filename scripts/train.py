import logging
from pathlib import Path
from dataclasses import asdict, dataclass
import warnings
import os
from typing import Any
from contextlib import nullcontext
import secrets

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR
from torch.utils.data import DataLoader

from tensordict.nn import TensorDictModule

from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAUROC

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

from muonly.data.transforms import Compose
from muonly.utils.optim import get_parameter_groups
from muonly.utils.optim import configure_lr_scheduler
from muonly.callbacks import ModelCheckpoint
from muonly.callbacks import MemoryTracker
from muonly.callbacks import CUDAMemoryTracker
from muonly.utils.reproducibility import set_seed

mh.style.use("CMS")

# FIXME: better way to set project root for hydra
os.environ["PROJECT_ROOT"] = str(Path(__file__).parents[1].resolve())

_logger = logging.getLogger(Path(__file__).name)
logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

for name in ["matplotlib", "PIL", "aim", "h5py", "filelock"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

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


@dataclass(repr=True)
class GlobalState:
    """Class to hold global state information such as current step and epoch."""

    step: int = 0
    epoch: int = 0
    val_loss: float = float("inf")
    val_auroc: float = 0.0
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_auroc: float = 0.0

    def state_dict(self):
        return asdict(self)

    def update_epoch(self, loss: float, auroc: float):
        """Update the best loss and AUROC if the current values are better."""
        self.val_loss = loss
        self.val_auroc = auroc

        if loss < self.best_val_loss:
            self.best_epoch = self.epoch
            self.best_val_loss = loss
            self.best_val_auroc = auroc

    def __str__(self):
        return (
            f"Epoch: {self.epoch}, Step: {self.step}, "
            f"Val Loss: {self.val_loss:.4f}, Val AUROC: {self.val_auroc:.4f} "
            f"(Best Epoch: {self.best_epoch}, Val Loss: {self.best_val_loss:.4f}, Val AUROC: {self.best_val_auroc:.4f})"
        )


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

    for batch in progress_bar:
        batch = batch.to(device)

        with amp_context:
            batch = model(batch)

            mask = batch["tracker_track_data_mask"]
            logits = batch["logits"][mask]
            target = batch["target"][mask]
            loss = criterion(input=logits, target=target)
            loss = loss.mean()

        loss.backward()
        clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=config.optim.max_grad_norm,
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


@torch.inference_mode()
def validate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    config: DictConfig,
    global_state: GlobalState,
    aim_run: aim.Run,
    amp_context: torch.autocast | nullcontext,
) -> dict[str, Any]:
    """ """
    # ---------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------
    model.eval()

    # ---------------------------------------------------------------------------
    # Metrics and histogram setup
    # ---------------------------------------------------------------------------

    loss_metric = MeanMetric()
    auroc_metric = BinaryAUROC()
    h_sig = Hist.new.Reg(40, 0, 1).Double()
    h_bkg = h_sig.copy()

    # ---------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------

    for batch in tqdm.rich.tqdm(
        data_loader, desc="Evaluating", disable=(not config.misc.tqdm)
    ):
        batch = batch.to(device)

        with amp_context:
            batch = model(batch)

            mask = batch["tracker_track_data_mask"]
            logits = batch["logits"][mask]
            target = batch["target"][mask]
            preds = logits.sigmoid()
            loss = criterion(input=logits, target=target)

        loss = loss.float().cpu()
        preds = preds.float().cpu()
        target = target.float().cpu()

        loss_metric.update(loss)
        auroc_metric.update(preds=preds, target=target)
        h_sig.fill(preds[target == 1].numpy())
        h_bkg.fill(preds[target == 0].numpy())

    # ---------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------
    result = {}
    result["loss"] = loss_metric.compute().item()
    result["auroc"] = auroc_metric.compute().item()

    # NOTE:
    fig, ax = plt.subplots()
    h_bkg.plot(ax=ax, label="background", histtype="step", color="tab:blue")
    h_sig.plot(ax=ax, label="signal", histtype="step", color="tab:orange")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    result["dist_score"] = Image(fig)

    return result


def configure_model_in_keys(config: DictConfig) -> dict[str, str]:
    """Configure the input keys for the model based on the data configuration."""
    in_key_list = []
    if config.data.tracker_track:
        in_key_list += ["tracker_track", "tracker_track_data_mask"]
    if config.data.dt_segment:
        in_key_list += ["dt_segment", "dt_segment_data_mask"]
    if config.data.csc_segment:
        in_key_list += ["csc_segment", "csc_segment_data_mask"]
    if config.data.gem_segment:
        in_key_list += ["gem_segment", "gem_segment_data_mask"]
    if config.data.rpc_hit:
        in_key_list += ["rpc_hit", "rpc_hit_data_mask"]
    if config.data.gem_hit:
        in_key_list += ["gem_hit", "gem_hit_data_mask"]

    in_keys = {each: each for each in in_key_list}
    return in_keys


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
    run_dir = Path(config.paths.run_dir)

    memory_tracker = MemoryTracker(
        output_dir=run_dir,
    )

    cuda_memory_tracker = CUDAMemoryTracker(
        device=torch.device(config.torch.device),
        output_dir=run_dir,
    )

    # ---------------------------------------------------------------------------
    # Save config
    # ---------------------------------------------------------------------------
    with open(run_dir / "config.yaml", "w") as file:
        OmegaConf.save(config=config, f=file, resolve=True)

    aim_run.name = config.run
    aim_run["config"] = OmegaConf.to_container(config, resolve=True)

    # ---------------------------------------------------------------------------
    # PyTorch setup
    # ---------------------------------------------------------------------------
    torch.set_float32_matmul_precision(config.torch.float32_matmul_precision)

    if config.torch.num_threads is not None:
        torch.set_num_threads(config.torch.num_threads)

    if config.torch.num_interop_threads is not None:
        torch.set_num_interop_threads(config.torch.num_interop_threads)

    set_seed(config.torch.seed)

    device = torch.device(config.torch.device)
    _logger.info(f"{device=}")
    if device.type == "cuda":
        _logger.info(f"GPU Name: {torch.cuda.get_device_name(device)}")

    # ---------------------------------------------------------------------------
    # Instantiate model
    # ---------------------------------------------------------------------------
    model = instantiate(config.model)
    memory_tracker.track("model_instantiated")

    # summarize model
    model_statistics = torchinfo.summary(model=model, verbose=False)
    _logger.info(f"Model summary:\n{model_statistics}")

    with open(run_dir / "model-summary.txt", "w") as file:
        file.write(f"{model_statistics}\n\nModel architecture:\n{model}")

    # ---------------------------------------------------------------------------
    # Model compilation
    # FIXME: failed to compile the model with torch 2.10.0 on khu
    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # Wrap model with TensorDictModule to handle input and output keys
    # ---------------------------------------------------------------------------

    td_model = TensorDictModule(
        module=(compiled_model if compiled_model is not None else model),
        in_keys=configure_model_in_keys(config=config),
        out_keys=[
            "logits",
        ],
    )

    memory_tracker.track("model_tensor_dict")

    # ---------------------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------------------
    train_set = instantiate(config.dataset.dataset)(**config.dataset.train)
    _logger.info(f"{train_set=}")
    _logger.info(f"Number of training examples: {len(train_set)}")

    memory_tracker.track("train_set_instantiated")

    val_set = instantiate(config.dataset.dataset)(**config.dataset.val)
    _logger.info(f"{val_set=}")
    _logger.info(f"Number of validation examples: {len(val_set)}")

    memory_tracker.track("val_set_instantiated")

    # ---------------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------------
    preprocessing = {
        key: Compose(instantiate(value)) for key, value in config.preprocessing.items()
    }

    train_set.apply_(preprocessing)
    val_set.apply_(preprocessing)

    memory_tracker.track("preprocessing")

    # ---------------------------------------------------------------------------
    # Data loaders
    # ---------------------------------------------------------------------------
    pin_memory = config.data_loader.pin_memory and (device.type == "cuda")
    _logger.info(f"{pin_memory=}")

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.data_loader.batch_size,
        shuffle=True,
        num_workers=config.data_loader.num_workers,
        collate_fn=train_set.collate,
        pin_memory=pin_memory,
        drop_last=True,  # drop last to avoid issues stem from small non-representative batches during training
        generator=torch.Generator().manual_seed(
            config.torch.seed
        ),  # ensure reproducibility when shuffling
    )
    _logger.info(f"Number of training batches: {len(train_loader)}")

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=config.data_loader.eval_batch_size,
        shuffle=False,  # no shuffling for evaluation
        num_workers=config.data_loader.num_workers,
        collate_fn=val_set.collate,
        pin_memory=pin_memory,
        drop_last=False,  # we want to validate on all validation examples
    )
    _logger.info(f"Number of validation batches: {len(val_loader)}")

    # ---------------------------------------------------------------------------
    # criterion
    # ---------------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(config.optim.pos_weight),
        reduction="none",
    )

    # ---------------------------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------------------------
    optimizer = AdamW(
        params=get_parameter_groups(
            model=model,
            weight_decay=config.optim.weight_decay,
        ),
        lr=config.optim.lr,
        betas=(
            config.optim.beta1,
            config.optim.beta2,
        ),
    )
    _logger.info(f"{optimizer=}")

    # ---------------------------------------------------------------------------
    # Learning rate scheduler
    # ---------------------------------------------------------------------------
    lr_scheduler = configure_lr_scheduler(
        optimizer=optimizer,
        num_steps_per_epoch=len(train_loader),
        max_epochs=config.optim.max_epochs,
        max_lr=config.optim.lr,
        warmup_frac=config.optim.warmup.frac_steps,
        warmup_start_factor=config.optim.warmup.start_factor,
        annealing_eta_min_factor=config.optim.annealing.eta_min_factor,
    )

    # ---------------------------------------------------------------------------
    # GPU setup
    # ---------------------------------------------------------------------------
    _logger.info("Moving model and criterion to device...")

    cuda_memory_tracker.track("before_move_to_device")

    model = model.to(device)
    criterion = criterion.to(device)

    cuda_memory_tracker.track("after_move_to_device")

    # ---------------------------------------------------------------------------
    # Mixed precision setup
    # ---------------------------------------------------------------------------
    if config.torch.precision == "float32":
        _logger.info("Using full precision (float32) for training.")
        amp_context = nullcontext()
    elif config.torch.precision == "bfloat16":
        _logger.info("Using mixed precision (bfloat16) for training.")
        amp_context = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported precision: {config.torch.precision}")

    # ---------------------------------------------------------------------------
    # Global state setup
    # ---------------------------------------------------------------------------
    global_state = GlobalState()

    # ---------------------------------------------------------------------------
    # Checkpointing setup
    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------

    cuda_memory_tracker.track("before_training_loop")

    for epoch in tqdm.rich.trange(0, 1 + config.optim.max_epochs, desc="Epoch"):
        global_state.epoch = epoch

        if epoch >= 1:
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
            memory_tracker.track(f"epoch_{epoch:06d}_train")
            cuda_memory_tracker.track(f"epoch_{epoch:06d}_train")

        val_result = validate(
            model=td_model,
            criterion=criterion,
            data_loader=val_loader,
            device=device,
            config=config,
            global_state=global_state,
            aim_run=aim_run,
            amp_context=amp_context,
        )

        memory_tracker.track(f"epoch_{epoch:06d}_val")
        cuda_memory_tracker.track(f"epoch_{epoch:06d}_val")

        for key, value in val_result.items():
            aim_run.track(
                value=value,
                name=key,
                epoch=global_state.epoch,
                step=global_state.step,
                context={"subset": "val"},
            )
        plt.close("all")

        global_state.update_epoch(
            loss=val_result["loss"],
            auroc=val_result["auroc"],
        )

        _logger.info(global_state)

        model_checkpoint.step(metric=val_result)

    # ---------------------------------------------------------------------------
    # Final evaluation and cleanup
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # Load best checkpoint
    # ---------------------------------------------------------------------------

    best_checkpoint = torch.load(model_checkpoint.best_path)
    model.load_state_dict(best_checkpoint["model"])
    del best_checkpoint

    # ---------------------------------------------------------------------------
    # Run final evaluation on validation set with best model checkpoint
    # ---------------------------------------------------------------------------

    # TODO:


@hydra.main(
    config_path="../config",
    config_name="main",
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
                "PyTorch anomaly detection is enabled. This may significantly slow down training. Use with caution."
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

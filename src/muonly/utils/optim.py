import logging
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import Optimizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR


_logger = logging.getLogger(__name__)


__all__ = [
    "get_parameter_groups",
    "configure_optimizer",
    "configure_lr_scheduler",
]


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict]:
    """ """
    if weight_decay < 0.0 or weight_decay > 1.0:
        raise ValueError(
            f"Invalid weight_decay value: {weight_decay}. Expected a value between 0.0 and 1.0."
        )
    if weight_decay == 0.0:
        _logger.warning(
            "Weight decay is set to 0.0. All parameters will be included in the no_decay group."
        )
        return [{"params": list(model.parameters()), "weight_decay": 0.0}]

    decay_param_list = []
    no_decay_param_list = []

    for module_name, module in model.named_modules():
        param_name_list = []
        param_list = []
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            param_name_list.append(name)
            param_list.append(param)

        if len(param_list) == 0:
            _logger.debug(
                f"Module '{module_name}' of type '{type(module).__name__}' has no trainable parameters. Skipping this module."
            )
            continue

        if isinstance(module, (nn.Linear, _ConvNd)):
            decay_param_list.append(module.weight)
            if module.bias is not None:
                no_decay_param_list.append(module.bias)
        elif isinstance(module, nn.Embedding):
            decay_param_list.append(module.weight)
        elif isinstance(module, nn.RNNBase):
            for name, param in module.named_parameters():
                # https://github.com/pytorch/pytorch/blob/v2.10.0/torch/nn/modules/rnn.py#L192C32-L200
                if name.startswith("weight_"):
                    decay_param_list.append(param)
                elif name.startswith("bias_"):
                    no_decay_param_list.append(param)
                else:
                    raise ValueError(
                        f"Unexpected parameter name '{name}' in RNN module. Expected names to start with 'weight_' or 'bias_'."
                    )
        elif isinstance(module, nn.MultiheadAttention):
            for name, param in zip(param_name_list, param_list):
                if name.endswith("weight"):
                    decay_param_list.append(param)
                elif name.endswith("bias"):
                    no_decay_param_list.append(param)
                else:
                    raise ValueError(
                        f"Unexpected parameter name '{name}' in MultiheadAttention module. Expected names to end with 'weight' or 'bias'."
                    )
        elif isinstance(module, (_BatchNorm, nn.LayerNorm)):
            no_decay_param_list += param_list
        else:
            for param_name, param in zip(param_name_list, param_list):
                _logger.warning(
                    f"Module '{module_name}' of type '{type(module).__name__}' is not explicitly handled. Parameter '{param_name}' will be included in the weight decay group by default."
                )
                decay_param_list.append(param)

    # TODO: Add a check to ensure that all parameters are included in either decay_param_list or no_decay_param_list, and that there are no duplicates.

    return [
        {"params": decay_param_list, "weight_decay": weight_decay},
        {"params": no_decay_param_list, "weight_decay": 0.0},
    ]


def configure_optimizer(model, lr, weight_decay, beta1, beta2):
    return AdamW(
        params=get_parameter_groups(
            model=model,
            weight_decay=weight_decay,
        ),
        lr=lr,
        betas=(
            beta1,
            beta2,
        ),
    )


def configure_lr_scheduler(
    optimizer: Optimizer,
    num_steps_per_epoch: int,
    max_epochs: int,
    max_lr: float,
    warmup_frac: float,
    warmup_start_factor: float,
    annealing_eta_min_factor: float,
) -> SequentialLR:
    """ """
    total_steps = num_steps_per_epoch * max_epochs
    warmup_steps = int(warmup_frac * total_steps)
    annealing_steps = total_steps - warmup_steps

    warmup = LinearLR(
        optimizer=optimizer,
        start_factor=warmup_start_factor,
        total_iters=warmup_steps,
    )

    annealing_eta_min = max_lr * annealing_eta_min_factor
    annealing = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=annealing_steps,
        eta_min=annealing_eta_min,
    )

    lr_scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[
            warmup,
            annealing,
        ],
        milestones=[
            warmup_steps,
        ],
    )

    return lr_scheduler

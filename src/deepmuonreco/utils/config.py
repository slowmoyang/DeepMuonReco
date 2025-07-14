from collections import OrderedDict
from typing import Any
from tensordict.nn import TensorDictModule, TensorDictSequential
from omegaconf import DictConfig
from lightning.pytorch.trainer.states import RunningStage
from torchmetrics import MetricCollection
from hydra.utils import instantiate
from ..metrics import TensorDictMetricWrapper


__all__ = [
    'build_tensordictmodule',
    'build_tensordictsequential',
    'build_metric_collection',
]


def normalize_keys(keys: list[str | list[str]]) -> list[str | tuple[str, ...]]:
    """
    Normalize the keys to ensure they are either strings or tuples of strings.
    """
    return [
        each if isinstance(each, str) else tuple(each)
        for each in keys
    ]


def build_tensordictmodule(
    config: DictConfig,
    instantiation: bool = True,
) -> TensorDictModule:
    """
    """
    if instantiation:
        config = instantiate(config)

    in_keys = normalize_keys(config.in_keys)
    out_keys = normalize_keys(config.out_keys)
    inplace = config.get('inplace', True)

    return TensorDictModule(
        module=config.module,
        in_keys=in_keys,
        out_keys=out_keys,
        inplace=inplace,
    )


def build_tensordictsequential(
    config: DictConfig,
    instantiation: bool = True,
) -> TensorDictSequential:
    """
    """
    if instantiation:
        config = instantiate(config)


    modules = OrderedDict()
    for each in config.modules:
        name = each['name']
        try:
            modules[name] = build_tensordictmodule(each)
        except Exception as error:
            raise ValueError(f"Error building module {name}: {error}")

    selected_out_keys = config.get('selected_out_keys', None)
    if selected_out_keys is not None:
        selected_out_keys = list(selected_out_keys)
    return TensorDictSequential(modules, selected_out_keys=selected_out_keys)


def build_metric_collection(
    config,
    stage: RunningStage,
    instantiation: bool = True,
) -> MetricCollection:
    """
    """
    if instantiation:
        config = instantiate(config)

    metric_dict: dict[str, TensorDictMetricWrapper] = {
        key: TensorDictMetricWrapper(**value)
        for key, value in config.metrics.items()
    }

    return MetricCollection(
        metrics=metric_dict, # type: ignore
        prefix=f'{stage.value}_',
        compute_groups=config.get('compute_groups', False),
    )


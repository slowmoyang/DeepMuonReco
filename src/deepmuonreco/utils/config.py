from collections import OrderedDict
from logging import getLogger
from tensordict.nn import TensorDictModule, TensorDictSequential
from omegaconf import DictConfig, ListConfig
from hydra.utils import instantiate
from torchmetrics import MetricCollection
from ..metrics import TensorDictWrapper


_logger = getLogger(__name__)


__all__ = [
    'build_tensordictmodule',
    'build_tensordictsequential',
    'build_metric_collection',
]



def normalize_keys(
    keys: list[str | list[str]],
) -> list[str | tuple[str, ...]]:
    """
    """
    output: list[str | tuple[str, ...]] = []
    for each in keys:
        if isinstance(each, str):
            output.append(each)
        elif isinstance(each, (list, ListConfig)):
            output.append(tuple(each))
        else:
            raise ValueError(f'{type(each)=}')
    return output


def build_tensordictmodule(
    config: DictConfig,
    instantiation: bool = True,
) -> TensorDictModule:
    """
    """
    if instantiation:
        _logger.debug(f'Instantiating config: {config}')
        config = instantiate(config)

    in_keys = normalize_keys(config['in_keys'])
    out_keys = normalize_keys(config['out_keys'])
    inplace = config.get('inplace', True)

    _logger.debug(f'Building TensorDictModule with config: {config}')
    return TensorDictModule(
        module=config['module'],
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
        _logger.debug(f'Instantiating config: {config}')
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
    config: DictConfig,
    instantiation: bool = True,
) -> MetricCollection:
    """
    """
    if instantiation:
        _logger.debug(f'Instantiating config: {config}')
        config = instantiate(config)

    metrics: dict = {
        key: TensorDictWrapper(**value)
        for key, value in config.metrics.items()
    }

    prefix = config.get('prefix', None)
    postfix = config.get('postfix', None)
    compute_groups: bool | list[list[str]] = config.get('compute_groups', True)

    return MetricCollection(
        metrics=metrics,
        prefix=prefix,
        postfix=postfix,
        compute_groups=compute_groups,
    )

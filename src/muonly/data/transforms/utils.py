import logging
from hydra.utils import instantiate
from .compose import Compose

_logger = logging.getLogger(__name__)


def configure_preprocessing(config):
    """Configure the preprocessing steps based on the provided configuration."""

    required_keys = ["tracker_track", "dt_segment", "csc_segment"]
    optional_keys = ["gem_segment", "rpc_hit", "gem_hit"]

    preprocessing = {}
    for key in required_keys:
        _logger.info(f"Configuring preprocessing for {key}...")
        preprocessing[key] = Compose(instantiate(config[key]["preprocessing"]))

    for key in optional_keys:
        if key not in config:
            _logger.info(f"Skipping preprocessing for {key} as it is not present in the configuration.")
            continue

        if config[key] is None:
            _logger.info(f"Skipping preprocessing for {key} as it is set to None in the configuration.")
            continue

        _logger.info(f"Configuring preprocessing for {key}...")
        preprocessing[key] = Compose(instantiate(config[key]["preprocessing"]))
    return preprocessing

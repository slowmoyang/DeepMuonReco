from hydra.utils import instantiate
from .compose import Compose


def configure_preprocessing(config):
    """Configure the preprocessing steps based on the provided configuration."""

    required_keys = ["tracker_track", "dt_segment", "csc_segment"]
    optional_keys = ["gem_segment", "rpc_hit", "gem_hit"]

    preprocessing = {
        key: Compose(instantiate(config[key]["preprocessing"])) for key in required_keys
    }
    for key in optional_keys:
        if key not in config or config[key] is None:
            continue
        preprocessing[key] = Compose(instantiate(config[key]["preprocessing"]))
    return preprocessing

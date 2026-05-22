import logging
from omegaconf import DictConfig


__all__ = [
    "configure_model_in_keys",
]

_logger = logging.getLogger(__name__)


def configure_model_in_keys(config: DictConfig) -> dict[str, str]:
    """Configure the input keys for the model based on the data configuration."""
    in_key_list = []
    if config.data.tracker_track:
        in_key_list += ["tracker_track", "tracker_track_data_mask"]
    else:
        raise ValueError(
            "Tracker track data is required for the model. Please enable it in the config."
        )

    if config.data.dt_segment:
        in_key_list += ["dt_segment", "dt_segment_data_mask"]
    else:
        _logger.warning(
            "DT segment data is disabled. Make sure this is intentional and that the model can handle the absence of this data."
        )

    if config.data.csc_segment:
        in_key_list += ["csc_segment", "csc_segment_data_mask"]
    else:
        _logger.warning(
            "CSC segment data is disabled. Make sure this is intentional and that the model can handle the absence of this data."
        )

    if config.data.gem_segment:
        in_key_list += ["gem_segment", "gem_segment_data_mask"]
    else:
        _logger.info(
            "GEM segment data is disabled. This is fine as long as the model is designed to work without it."
        )

    if config.data.rpc_hit:
        in_key_list += ["rpc_hit", "rpc_hit_data_mask"]
    else:
        _logger.info(
            "RPC hit data is disabled. This is fine as long as the model is designed to work without it."
        )

    if config.data.gem_hit:
        in_key_list += ["gem_hit", "gem_hit_data_mask"]
    else:
        _logger.info(
            "GEM hit data is disabled. This is fine as long as the model is designed to work without it."
        )

    in_keys = {each: each for each in in_key_list}
    return in_keys

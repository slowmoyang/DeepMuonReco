"""
get parameters required for data preprocessing and save them to a file.
"""

from omegaconf import OmegaConf


data_config = OmegaConf.load("../config/data/mu2030pu.yaml")
dataset_config = OmegaConf.load("../config/dataset/default.yaml")

print(f"{data_config=}")
print(f"{dataset_config=}")

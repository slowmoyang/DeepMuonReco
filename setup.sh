#!/usr/bin/env bash
export PROJECT_ROOT=$(realpath $(dirname ${BASH_SOURCE}))
export PYTHONPATH=${PROJECT_ROOT}/src:${PYTHONPATH}
micromamba activate deepmuonreco-py312

#!/usr/bin/env bash
export PROJECT_PREFIX=$(realpath $(dirname ${BASH_SOURCE}))
export PYTHONPATH=${PROJECT_PREFIX}/src:${PYTHONPATH}
micromamba activate deepmuonreco-py312

#!/bin/bash

source ~/.bashrc
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
source scripts/activate_conda_env.sh

python scripts/preprocess_esmfold.py

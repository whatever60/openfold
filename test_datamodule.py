import argparse
import logging
import os
import random
import sys
import time
import json

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin, DDPPlugin
from pytorch_lightning.plugins.environments import SLURMEnvironment
import torch
from flatten_dict import flatten, unflatten
import yaml
import ml_collections as mlc


from openfold.config import model_config
from openfold.data.data_modules import (
    OpenFoldDataModule,
    DummyDataLoader,
)
from openfold.model.model import AlphaFold
from openfold.model.torchscript import script_preset_
from openfold.np import residue_constants
from openfold.utils.argparse import remove_arguments
from openfold.utils.callbacks import (
    EarlyStoppingVerbose,
)
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.loss import AlphaFoldLoss, lddt_ca
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils.seed import seed_everything
from openfold.utils.superimposition import superimpose
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
)
from openfold.utils.import_weights import import_jax_weights_

from scripts.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint,
    get_global_step_from_zero_checkpoint,
)

from openfold.utils.logger import PerformanceLoggingCallback
from openfold.utils.config_check import enforce_config_constraints

from train_openfold import OpenFoldWrapper, enforce_arg_constrains


args = dict(
    # train
    train_data_dir="/scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files",
    train_alignment_dir="/scratch/00946/zzhang/data/openfold/ls6-tacc/alignment_db",
    alignment_index_path="/scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/chain_lists/duplicated_super_fix.index",
    obsolete_pdbs_file_path="/scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/obsolete.dat",
    use_small_bfd=False,
    train_filter_path=None,
    train_chain_data_cache_path="/scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/prot_data_cache.json",
    # template
    template_mmcif_dir="/scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files",
    max_template_date="2021-10-01",
    template_release_dates_cache_path="/scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/mmcif_cache.json",
    kalign_binary_path="/usr/bin/kalign",
    # validation
    val_data_dir="/scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/val_set/data",
    val_alignment_dir="/scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/val_set/alignments",
    # model
    # {initial_training/fintuning/fintuning_no_template/model_1.1/model_1.2/model_1.1.1/model_1.1.2/model_1.2.1/model_1.2.2/model_1.2.3}
    config_stage="initial_training",
    # {ptm/None}
    config_ptm=False,
    # {train/inference_long_seq/None}
    config_mode="train",
    # {low_prec/None}
    config_lowprec=False,
    script_modules=False,
    # ditillation
    distillation_data_dir=None,
    distillation_alignment_dir=None,
    distillation_filter_path=None,
    distillation_alignment_index_path=None,
    _distillation_structure_index_path=None,
    distillation_chain_data_cache_path=None,
    # logging
    log_lr=True,
    checkpoint_every_epoch=True,
    output_dir="train_baseline/test",
    log_performance=False,
    wandb=True,
    wandb_entity="openfold",
    wandb_project="single_sequence_yiming",
    experiment_name="no_msa_no_template",
    wandb_id=None,
    # parallel
    gpus=3,
    num_nodes=1,
    replace_sampler_ddp=True,
    deepspeed_config_path="deepspeed_config.json",
    # trainer
    seed=42,
    train_epoch_len=126000,
    accumulate_grad_batches=3,
    num_sanity_val_steps=0,
    reload_dataloaders_every_n_epochs=1,
    resume_from_ckpt=None,
    resume_model_weights_only=False,
    # early stopping
    early_stopping=False,
    min_delta=0,
    patience=3,
)


enforce_arg_constrains(args)

# load config
config = {}


def update_config(old: dict, new: dict) -> dict:
    return unflatten({**flatten(config), **flatten(new)})


with open(f"configs/{args['config_stage']}.json") as f:
    config = update_config(config, json.load(f))
if args["config_ptm"]:
    with open(f"configs/ptm.json") as f:
        config = update_config(config, json.load(f))
with open(f"configs/{args['config_mode']}.json") as f:
    config = update_config(config, json.load(f))
if args["config_lowprec"]:
    with open(f"configs/low_prec.json") as f:
        config = update_config(config, json.load(f))

enforce_config_constraints(config)

if args["seed"] is not None:
    seed_everything(args["seed"])

# for the sake of testing
config["data"]["data_module"]["data_loaders"] = {
    "batch_size": 1,
    "num_workers": 8,
    "pin_memory": True,
}
config["globals"]["chunk_size"] = None


# training mode (train_data_dir passed)
data_module = OpenFoldDataModule(
    config=mlc.ConfigDict(config["data"]),
    template_mmcif_dir=args["template_mmcif_dir"],
    max_template_date=args["max_template_date"],
    train_data_dir=args["train_data_dir"],
    train_alignment_dir=args["train_alignment_dir"],
    train_chain_data_cache_path=args["train_chain_data_cache_path"],
    distillation_data_dir=args["distillation_data_dir"],
    distillation_alignment_dir=args["distillation_alignment_dir"],
    distillation_chain_data_cache_path=args["distillation_chain_data_cache_path"],
    val_data_dir=args["val_data_dir"],
    val_alignment_dir=args["val_alignment_dir"],
    kalign_binary_path=args["kalign_binary_path"],
    train_filter_path=args["train_filter_path"],
    distillation_filter_path=args["distillation_filter_path"],
    obsolete_pdbs_file_path=args["obsolete_pdbs_file_path"],
    template_release_dates_cache_path=args["template_release_dates_cache_path"],
    batch_seed=args["seed"],
    train_epoch_len=args["train_epoch_len"],
    _distillation_structure_index_path=args["_distillation_structure_index_path"],
    alignment_index_path=args["alignment_index_path"],
    distillation_alignment_index_path=args["distillation_alignment_index_path"],
    predict_data_dir=None,
    predict_alignment_dir=None,
    # **vars(args)
)

data_module.prepare_data()
data_module.setup()

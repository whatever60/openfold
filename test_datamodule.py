"""
Test data loading code. Make sure that for single sequence model, the model has the same output with different Dataset implementation.
"""

import json
from functools import partial
from time import perf_counter


from pytorch_lightning import seed_everything
import torch
from flatten_dict import flatten, unflatten
import ml_collections as mlc

from openfold.data.data_modules import OpenFoldDataModule, DummyDataLoader
from openfold.utils.config_check import enforce_config_constraints
from train_openfold import OpenFoldWrapper, enforce_arg_constrains


def load_args_msa():
    args = dict(
        # train
        train_data_dir="/scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files",
        # train_data_dir="/work/09101/whatever/data/example/alphafold",
        # train_data_dir="/work/09101/whatever/data/example/esmfold/001",
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
        # config
        config_stage="initial_training",  # {initial_training/fintuning/fintuning_no_template/model_1.1/model_1.2/model_1.1.1/model_1.1.2/model_1.2.1/model_1.2.2/model_1.2.3}
        config_ptm=False,  # {ptm/None}
        config_mode="train",  # {train/inference_long_seq/None}
        config_lowprec=False,  # {low_prec/None}
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
        train_epoch_len=1260,  # for the sake of testing
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
    return args


def load_args_baseline():
    args = load_args_msa()
    args["train_alignment_dir"] = "/scratch/09101/whatever/data/openfold/alignment_db"
    args[
        "alignment_index_path"
    ] = "/scratch/09101/whatever/data/openfold/alignment_db/duplicated_super_fix.index"
    args["val_alignment_dir"] = "/scratch/09101/whatever/data/openfold/val_set"
    enforce_arg_constrains(args)
    return args


def load_args_baseline_simple():
    args = load_args_msa()
    args["train_alignment_dir"] = None
    args["alignment_index_path"] = None
    args["val_alignment_dir"] = None
    args["template_mmcif_dir"] = None
    args["max_template_date"] = None
    args["template_release_dates_cache_path"] = None
    args["kalign_binary_path"] = None
    enforce_arg_constrains(args)
    return args


def load_args_esmfold_example():
    args = load_args_baseline()
    args["train_data_dir"] = "/work/09101/whatever/data/example/esmfold/001"
    args["alignment_index_path"] = "/work/09101/whatever/data/example/esmfold.index"
    args["obsolete_pdbs_file_path"] = None
    args[
        "train_chain_data_cache_path"
    ] = "/work/09101/whatever/data/example/esmfold_cache.json"
    enforce_arg_constrains(args)
    return args


def load_config(args):
    # load config
    with open(f"configs/base.json") as f:
        config = json.load(f)
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
    # for the sake of testing
    config["data"]["data_module"]["data_loaders"] = {
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": False,
    }
    config["globals"]["chunk_size"] = None
    enforce_config_constraints(config)
    return config


def update_config(old: dict, new: dict) -> dict:
    return unflatten({**flatten(old), **flatten(new)})


def get_datamodule_generator(args, config):
    return partial(
        OpenFoldDataModule,
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
    )


def test_openfold_simple():
    """Compare OpenFoldSingleDatset with OpenFoldSimpleSingleDataset under single-sequence setting.

    For OpenFoldSingleDataset, there is still some code remaining to process alignment
    data path, and thus dummy alignment data is required.
    For OpenFoldSimpleSingleDataset, all alignment related stuff are removed. The code
    will take care of generate dummy alignment to make samples have the same structure as OpenFoldSingleData.

    Therefore, sampling from these two datasets under single-sequence setting should give
    exact same results.
    """

    args = load_args_baseline()
    config = load_config(args)
    gen_datamodule = get_datamodule_generator(args, config)

    # model_module = OpenFoldWrapper(mlc.ConfigDict(config)).eval()

    # just training mode (train_data_dir passed)
    seed_everything(args["seed"])

    # OpenFoldSingleDataset
    t = perf_counter()
    datamodule_baseline = gen_datamodule(simple=False)
    datamodule_baseline.prepare_data()
    datamodule_baseline.setup()
    dataloader = datamodule_baseline.train_dataloader()
    print("Default dataset, loading:", perf_counter() - t)
    t = perf_counter()
    sample_baseline = next(iter(dataloader))
    print("Default dataset, sampling:", perf_counter() - t)
    # with torch.no_grad():
    #     output_baseline = model_module(sample_baseline)

    # OpenFoldSimpleSingleDataset
    seed_everything(args["seed"])
    t = perf_counter()
    datamodule_baseline_simple = gen_datamodule(
        simple=True,
        chain_index_path_train="/scratch/09101/whatever/data/openfold/chain_index_train.json",
        chain_index_path_val="/scratch/09101/whatever/data/openfold/chain_index_val.json",
    )
    datamodule_baseline_simple.prepare_data()
    datamodule_baseline_simple.setup()
    dataloader = datamodule_baseline_simple.train_dataloader()
    print("Simple dataset, loading:", perf_counter() - t)
    t = perf_counter()
    sample_baseline_simple = next(iter(dataloader))
    print("Simple dataset:", perf_counter() - t)
    # with torch.no_grad():
    #     output_baseline_simple = model_module(sample_baseline_simple)

    assert all(isinstance(v, torch.Tensor) for v in sample_baseline.values())
    assert sample_baseline.keys() == sample_baseline_simple.keys()
    assert all(
        (sample_baseline[k] == sample_baseline_simple[k]).all() for k in sample_baseline
    )


def test_esmfold_example():
    args = load_args_esmfold_example()
    config = load_config(args)
    gen_datamodule = get_datamodule_generator(args, config)

    # OpenFoldSingleDataset
    datamodule_baseline = gen_datamodule(simple=False)
    datamodule_baseline.prepare_data()
    datamodule_baseline.setup()
    dataloader = datamodule_baseline.train_dataloader()
    sample_baseline = next(iter(dataloader))
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    # test_esmfold_example()
    args = load_args_esmfold_example()
    config = load_config(args)
    gen_datamodule = get_datamodule_generator(args, config)

    # OpenFoldSingleDataset
    datamodule_baseline = gen_datamodule(simple=False)
    datamodule_baseline.prepare_data()
    datamodule_baseline.setup()
    dataloader = datamodule_baseline.train_dataloader()
    sample_baseline = next(iter(dataloader))

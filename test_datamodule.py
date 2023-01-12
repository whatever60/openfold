"""
This file consists of code that handles arguments of OpenFold components, especially various paths.
All paths here are specfic to Yiming's use case on TACC server.
"""

import json
from functools import partial
from time import perf_counter
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from flatten_dict import flatten, unflatten
import ml_collections as mlc
from tqdm.auto import tqdm

from openfold.data.data_modules import OpenFoldDataModule, DummyDataLoader
from openfold.utils.config_check import enforce_config_constraints
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.tensor_utils import tensor_tree_map
from openfold import config as openfold_config

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


def load_args_baseline_casp15():
    args = load_args_msa()
    args["train_alignment_dir"] = "/scratch/09101/whatever/data/openfold/alignment_db"
    args[
        "alignment_index_path"
    ] = "/scratch/09101/whatever/data/openfold/alignment_db/duplicated_super_fix.index"
    args["val_data_dir"] = "/scratch/09101/whatever/data/casp15/data"
    args["val_alignment_dir"] = "/scratch/09101/whatever/data/casp15/alignment"
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
    """deprecated
    """
    args = load_args_baseline()
    args["train_data_dir"] = "/work/09101/whatever/data/example/esmfold/001"
    args["alignment_index_path"] = "/work/09101/whatever/data/example/esmfold.index"
    args["obsolete_pdbs_file_path"] = None
    args[
        "train_chain_data_cache_path"
    ] = "/work/09101/whatever/data/example/esmfold_cache.json"
    enforce_arg_constrains(args)
    return args


def fix_validation_arg(args, baseline: bool) -> None:
    """On 1/12/2023, a critical mistake was found in the setting of validation paths.
    The example script of training OpenFold uses a validation set that seems to be 
    constructed by randomly selecting 100 from the training set. So validation metrics 
    are incorrect and essentially training metrics.
    The correct validation set (180 CAMEO proteins) is in another directory.
    """

    args["val_data_dir"] = "/scratch/00946/zzhang/data/openfold/cameo/mmcif_files"
    if baseline is True:
        args["val_alignment_dir"]= "/scratch/09101/whatever/data/openfold/cameo_alignments"
    else:
        args["val_alignment_dir"]= "/scratch/00946/zzhang/data/openfold/cameo/alignments"


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
    config = mlc.ConfigDict(config["data"])
    return partial(
        OpenFoldDataModule,
        config=config,
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
    fix_validation_arg(args, baseline=True)
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
    fix_validation_arg(args, baseline=True)
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


def get_model_basename(model_path):
    return os.path.splitext(os.path.basename(os.path.normpath(model_path)))[0]


def load_af2_checkpoint(model_version: str = "model_2_ptm") -> pl.LightningDataModule:
    args = load_args_msa()
    fix_validation_arg(args, baseline=False)
    if model_version == "model_2_ptm":
        args["config_stage"] = "model_1.1.2"  # this correspond to model_2
        args["config_ptm"] = True
        config_new = load_config(args)
        config = config_new  # must use tricks to decrease memory usage
        config = openfold_config.model_config("model_2_ptm")
        print(
            {
                k: (flatten(config_new)[k], flatten(config.to_dict())[k])
                for k in flatten(config.to_dict())
                if flatten(config_new)[k] != flatten(config.to_dict())[k]
            }
        )
    else:
        raise NotImplementedError
    path = f"params/af2/params_{model_version}.npz"
    model_module = OpenFoldWrapper(mlc.ConfigDict(config)).cuda().eval()
    import_jax_weights_(model_module.model, path, version=model_version)
    return model_module


def load_of_baseline_checkpoint(epoch: int) -> pl.LightningModule:
    args = load_args_baseline()
    fix_validation_arg(args, baseline=True)
    config = load_config(args)
    config["globals"]["blocks_per_ckpt"] = None
    config["globals"]["chunk_size"] = 4
    config["model"]["extra_msa"]["enabled"] = False
    config["model"]["template"]["enabled"] = False
    if epoch == 0:
        path = "2vrl7q7l/checkpoints/0-3499.ckpt"
    elif epoch == 1:
        path = "2vrl7q7l/checkpoints/1-6999.ckpt"
    elif epoch == 2:
        path = "2vrl7q7l/checkpoints/2-10499.ckpt"
    elif epoch == 3:
        path = "ym2z0d6k/checkpoints/3-13999.ckpt"
    elif epoch == 4:
        path = "ym2z0d6k/checkpoints/4-17499.ckpt"
    elif epoch == 5:
        path = "ym2z0d6k/checkpoints/5-20999.ckpt"
    elif epoch == 6:
        path = "1ru2ahft/checkpoints/6-24499.ckpt"
    elif epoch == 7:
        path = "1ru2ahft/checkpoints/7-27999.ckpt"
    elif epoch == 8:
        path = "1ru2ahft/checkpoints/8-31499.ckpt"
    elif epoch == 9:
        path = "39o3w69r/checkpoints/9-34999.ckpt"
    elif epoch == 10:
        path = "39o3w69r/checkpoints/10-38499.ckpt"
    elif epoch == 11:
        path = "39o3w69r/checkpoints/11-41999.ckpt"
    elif epoch == 12:
        path = "3rjv33wo/checkpoints/12-45499.ckpt"
    elif epoch == 13:
        path = "3rjv33wo/checkpoints/13-48999.ckpt"
    elif epoch == 14:
        path = "3rjv33wo/checkpoints/14-52499.ckpt"
    elif epoch == 15:
        path = "b7i6lrtn/checkpoints/15-55999.ckpt"
    elif epoch == 16:
        path = "b7i6lrtn/checkpoints/16-59499.ckpt"
    elif epoch == 17:
        path = "1rur4vrq/checkpoints/17-62999.ckpt"
    elif epoch == 18:
        path = "1rur4vrq/checkpoints/18-66499.ckpt"
    elif epoch == 19:
        path = "1rur4vrq/checkpoints/19-69999.ckpt"
    elif epoch == 20:
        path = "oynkb1xe/checkpoints/20-73499.ckpt"
    elif epoch == 21:
        path = "oynkb1xe/checkpoints/21-76999.ckpt"
    elif epoch == 22:
        path = "oynkb1xe/checkpoints/22-80499.ckpt"
    elif epoch == 23:
        path = "2hvbagos/checkpoints/23-83999.ckpt"
    elif epoch == 24:
        path = "2hvbagos/checkpoints/24-87499.ckpt"
    elif epoch == 25:
        path = "2hvbagos/checkpoints/25-90999.ckpt"
    elif epoch == 26:
        path = "ncxgis53/checkpoints/26-94499.ckpt"
    elif epoch == 27:
        path = "ncxgis53/checkpoints/27-97999.ckpt"
    elif epoch == 28:
        path = "ncxgis53/checkpoints/28-101499.ckpt"
    elif epoch == 29:
        path = "2b86jtg8/checkpoints/29-104999.ckpt"
    elif epoch == 30:
        path = "2b86jtg8/checkpoints/30-108499.ckpt"
    elif epoch == 31:
        path = "2b86jtg8/checkpoints/31-111999.ckpt"
    elif epoch == 32:
        path = "3sa33b9m/checkpoints/32-115499.ckpt"
    elif epoch == 33:
        path = "3sa33b9m/checkpoints/33-118999.ckpt"
    elif epoch == 34:
        path = "3sa33b9m/checkpoints/34-122499.ckpt"
    elif epoch == 35:
        path = "2afpx0hx/checkpoints/35-125999.ckpt"
    elif epoch == 36:
        path = "2afpx0hx/checkpoints/36-129499.ckpt"
    elif epoch == 37:
        path = "2afpx0hx/checkpoints/37-132999.ckpt"
    elif epoch == 38:
        path = "fzowrxua/checkpoints/38-136499.ckpt"
    elif epoch == 39:
        path = "fzowrxua/checkpoints/39-139999.ckpt"
    elif epoch == 40:
        path = "fzowrxua/checkpoints/40-143499.ckpt"
    elif epoch == 41:
        path = "2zkz2gtf/checkpoints/41-146999.ckpt"
    elif epoch == 42:
        path = "2zkz2gtf/checkpoints/42-150499.ckpt"
    elif epoch == 43:
        path = "2zkz2gtf/checkpoints/43-153999.ckpt"
    elif epoch == 44:
        path = "1herl4j4/checkpoints/44-157499.ckpt"
    elif epoch == 45:
        path = "1herl4j4/checkpoints/45-160999.ckpt"
    elif epoch == 46:
        path = "1herl4j4/checkpoints/46-164499.ckpt"
    elif epoch == 47:
        path = "1pbccji9/checkpoints/47-167999.ckpt"
    elif epoch == 48:
        path = "1pbccji9/checkpoints/48-171499.ckpt"
    elif epoch == 49:
        path = "1pbccji9/checkpoints/49-174999.ckpt"
    elif epoch == 50:
        path = "tn1tdqyg/checkpoints/50-178499.ckpt"
    elif epoch == 51:
        path = "tn1tdqyg/checkpoints/51-181999.ckpt"
    elif epoch == 52:
        path = "tn1tdqyg/checkpoints/52-185499.ckpt"
    elif epoch == 53:
        path = "37ufgar3/checkpoints/53-188999.ckpt"
    else:
        raise NotImplementedError

    num_steps = int(path.split(".")[0].split("-")[1]) + 1
    d = torch.load(
        f"train_baseline/test/single_sequence_yiming/{path}/global_step{num_steps}/mp_rank_00_model_states.pt"
    )
    model_module = OpenFoldWrapper(mlc.ConfigDict(config))
    model_module.model.load_state_dict(d["ema"]["params"])
    model_module.cuda().eval()
    # trainer = pl.Trainer()
    # trainer._restore_modules_and_callbacks(model_module, f"train_baseline/test/single_sequence_yiming/{path}")
    # trainer.model
    # model_module = (
    #     OpenFoldWrapper
    #     .load_from_checkpoint(
    #         f"train_baseline/test/single_sequence_yiming/{path}/latest", config=mlc.ConfigDict(config)
    #     )
    #     .cuda()
    #     .eval()
    # )

    # if model_module.cached_weights is None:
    #     # model.state_dict() contains references to model weights rather
    #     # than copies. Therefore, we need to clone them before calling
    #     # load_state_dict().
    #     clone_param = lambda t: t.detach().clone()
    #     model_module.cached_weights = tensor_tree_map(
    #         clone_param, model_module.model.state_dict()
    #     )
    #     model_module.model.load_state_dict(model_module.ema.state_dict()["params"])
    return model_module


def load_val_dataloader(msa: bool = True) -> DataLoader:
    if msa is True:
        args = load_args_msa()
        fix_validation_arg(args, baseline=False)
        config = load_config(args)
    else:
        args = load_args_baseline()
        fix_validation_arg(args, baseline=True)
        config = load_config(args)
        config["model"]["extra_msa"]["enabled"] = False
        config["model"]["template"]["enabled"] = False
    gen_datamodule = get_datamodule_generator(args, config)
    datamodule_baseline = gen_datamodule(simple=False)
    datamodule_baseline.prepare_data()
    datamodule_baseline.setup()
    return datamodule_baseline.val_dataloader()


def load_val_casp15_dataloader() -> DataLoader:
    args = load_args_baseline_casp15()
    fix_validation_arg(args, baseline=True)
    config = load_config(args)
    config["model"]["extra_msa"]["enabled"] = False
    config["model"]["template"]["enabled"] = False
    gen_datamodule = get_datamodule_generator(args, config)
    datamodule_baseline = gen_datamodule(simple=False)
    datamodule_baseline.prepare_data()
    datamodule_baseline.setup()
    return datamodule_baseline.val_dataloader()


def cal_validation_res(
    model_module: pl.LightningModule, dataloader_val: DataLoader
) -> pd.DataFrame:
    # At the start of validation, load the EMA weights
    # Don't run this on AlphaFold2 weight.
    # if model_module.cached_weights is None:
    #     # model.state_dict() contains references to model weights rather
    #     # than copies. Therefore, we need to clone them before calling
    #     # load_state_dict().
    #     clone_param = lambda t: t.detach().clone()
    #     model_module.cached_weights = tensor_tree_map(
    #         clone_param, model_module.model.state_dict()
    #     )
    #     model_module.model.load_state_dict(
    #         model_module.ema.state_dict()["params"]
    #     )

    sequences = []
    # num_msas = []
    all_atom_positions = []
    final_atom_positions = []
    plddts = []
    tms = []
    paes = []
    losses = []
    metrics = []
    with torch.no_grad():
        for sample in tqdm(dataloader_val):
            batch = {k: v.cuda() for k, v in sample.items()}
            # Run the model
            outputs = model_module.model(batch)
            batch = tensor_tree_map(lambda t: t[..., -1], batch)

            # Compute loss and other metrics
            batch["use_clamped_fape"] = 0.0
            _, loss_breakdown = model_module.loss(
                outputs, batch, _return_breakdown=True, violation_breakdown=True
            )
            other_metrics = model_module._compute_validation_metrics(
                batch, outputs, superimposition_metrics=True
            )

            sequences.append(batch["aatype"][0].cpu().numpy())
            all_atom_positions.append(batch["all_atom_positions"][0].cpu().numpy())
            final_atom_positions.append(
                outputs["final_atom_positions"][0].cpu().numpy()
            )
            plddts.append(outputs["plddt"][0].cpu().numpy())
            tms.append(
                outputs["predicted_tm_score"].item()
                if "predicted_tm_score" in outputs
                else None
            )
            paes.append(
                outputs["predicted_aligned_error"][0].cpu().numpy()
                if "predicted_aligned_error" in outputs
                else None
            )
            metrics.append({k: v.item() for k, v in other_metrics.items()})
            losses.append({k: v.item() for k, v in loss_breakdown.items()})

    df = pd.DataFrame(
        {
            "sequence": sequences,
            "all_atom_positions": all_atom_positions,
            "final_atom_positions": final_atom_positions,
            "plddt": plddts,
            "tm": tms,
            "pae": paes,
            "loss": losses,
            "other_metrics": metrics,
        }
    )
    return df

    # with open("val_res/af2_model_2_ptm.pkl", "wb") as f:
    #     pickle.dump(
    #         {
    #             "final_atom_position": final_atom_positions,
    #             "metric": metrics,
    #             "loss": losses,
    #         },
    #         f,
    #     )


if __name__ == "__main__":
    # ==== test_esmfold_example() ====
    # args = load_args_esmfold_example()
    # config = load_config(args)
    # gen_datamodule = get_datamodule_generator(args, config)

    # ==== OpenFoldSingleDataset ====
    # datamodule_baseline = gen_datamodule(simple=False)
    # datamodule_baseline.prepare_data()
    # datamodule_baseline.setup()
    # dataloader = datamodule_baseline.train_dataloader()
    # sample_baseline = next(iter(dataloader))

    # ==== run validation for af2 ====
    # seed_everything(42)
    # df = cal_validation_res(load_af2_checkpoint(), load_val_dataloader())
    # df.to_pickle("val_res/af2_model_2_ptm.pkl")

    # ==== run CAMEO validation with openfold baseline ====
    for i in range(53, 52, -1):
        seed_everything(42)
        df = cal_validation_res(
            load_of_baseline_checkpoint(i), load_val_dataloader(msa=False)
        )
        df.to_pickle(f"val_res/of_baseline_{i}.pkl")

    # ==== run CASP15 4 cases with openfold baseline ====
    i = 53
    seed_everything(42)
    df = cal_validation_res(
        load_of_baseline_checkpoint(i), load_val_casp15_dataloader()
    )
    df.to_pickle(f"val_casp15_res/of_baseline_{i}.pkl")

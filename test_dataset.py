from functools import partial
import json

from flatten_dict import flatten, unflatten

from openfold.data import OpenFoldSingleDataset
from test_datamodule import args, update_config


template_mmcif_dir = (
    "/scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files"
)
max_template_date = "2021-10-01"
kalign_binary_path = "/usr/bin/kalign"
template_release_dates_cache_path = (
    "/scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/mmcif_cache.json"
)
obsolete_pdbs_file_path = (
    "/scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/obsolete.dat"
)


# train_data_dir = "/scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files"
train_data_dir = "/work/09101/whatever/data/example/alphafold"
train_alignment_dir = "/scratch/00946/zzhang/data/openfold/ls6-tacc/alignment_db"
train_chain_data_cache_path = (
    "/scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/prot_data_cache.json"
)
train_filter_path = None
alignment_index_path = "/scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/chain_lists/duplicated_super_fix.index"
alignment_index = None

if alignment_index_path is not None:
    with open(alignment_index_path, "r") as fp:
        alignment_index = json.load(fp)
config = {}

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

dataset_gen = partial(
    OpenFoldSingleDataset,
    template_mmcif_dir=template_mmcif_dir,
    max_template_date=max_template_date,
    config=config,
    kalign_binary_path=kalign_binary_path,
    template_release_dates_cache_path=template_release_dates_cache_path,
    obsolete_pdbs_file_path=obsolete_pdbs_file_path,
)

train_dataset = dataset_gen(
    data_dir=train_data_dir,
    chain_data_cache_path=train_chain_data_cache_path,
    alignment_dir=train_alignment_dir,
    filter_path=train_filter_path,
    max_template_hits=config.train.max_template_hits,
    shuffle_top_k_prefiltered=config.train.shuffle_top_k_prefiltered,
    treat_pdb_as_distillation=False,
    mode="train",
    alignment_index=alignment_index,
)

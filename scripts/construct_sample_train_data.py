import os
import random
import argparse
import subprocess
import json
import re

import numpy as np
import pandas as pd
from Bio.Align import parse
from tqdm.auto import tqdm


def get_templ_chains(hhr_file: str) -> list:
    return [
        f"{j.target.id.split('_')[0].lower()}_{j.target.id.split('_')[1]}"
        for j in parse(hhr_file, fmt="hhr")
    ]


def subsample(
    input_mmcif_dir,
    input_alignment_dir,
    val_data_dir,
    val_alignment_dir,
    output_dir,
    train_chain_data_cache_path,  # needs subsample
    template_release_dates_cache_path,  # needs subsample
    obsolete_pdbs_file_path,  # for replacing templates
    group_json,
    n_samples=100,
    n_templs=15,
):
    random.seed(7)
    # assuming chain files and directories are directly inside this directory
    all_samples = os.listdir(input_alignment_dir)
    # select 100 chains randomly from training set
    chains_train = random.sample(all_samples, n_samples)
    # all validation chains
    chains_val = os.listdir(val_alignment_dir)
    # for each selected chain, get the first few hits from its HHsearch result.
    templates_train = [
        get_templ_chains(f"{input_alignment_dir}/{i}/pdb70_hits.hhr")
        for i in chains_train
    ]
    templates_val = [
        get_templ_chains(f"{val_alignment_dir}/{i}/pdb70_hits.hhr") for i in chains_val
    ]
    sampled_idx = [
        random.sample(range(len(i)), n_templs)
        if n_templs < len(i) or n_templs == -1
        else list(range(len(i)))
        for i in templates_train + templates_val
    ]
    templates = [
        [c for i, c in enumerate(cs) if i in idx]
        for idx, cs in zip(sampled_idx, templates_train + templates_val)
    ]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/mmcif_files", exist_ok=True)
    os.makedirs(f"{output_dir}/alignments", exist_ok=True)
    os.makedirs(f"{output_dir}/alignment_db", exist_ok=True)
    os.makedirs(f"{output_dir}/val", exist_ok=True)
    os.makedirs(f"{output_dir}/val/mmcif_files", exist_ok=True)
    os.makedirs(f"{output_dir}/val/alignments", exist_ok=True)

    # Copy the alignment folder
    for chains, i_dir, o_dir, templs in zip(
        [chains_train, chains_val],
        [input_alignment_dir, val_alignment_dir],
        [f"{output_dir}/alignments", f"{output_dir}/val/alignments"],
        [sampled_idx[:len(chains_train)], sampled_idx[len(chains_train):]],
    ):
        for c, ts in zip(chains, templs):
            subprocess.run(["cp", "-rf", f"{i_dir}/{c}", o_dir], check=True)

            res = []
            num_hits = 0
            selected_hits = 0
            with open(f"{o_dir}/{c}/pdb70_hits.hhr") as f:
                for line in f:
                    # header
                    res.append(line)
                    if line.startswith(" No Hit"):
                        break
                for line in f:
                    # summary
                    if line == "\n":
                        res.append(line)
                        break
                    elif int(line.split()[0]) - 1 in ts:
                        num_hits += 1
                        selected_hits += 1
                        new_line = " "
                        remaining = line.split(maxsplit=1)[1]
                        length_digits = len(line) - len(remaining) - 2
                        new_line += (
                            str(selected_hits).rjust(length_digits) + " " + remaining
                        )
                        res.append(new_line)
                    else:
                        num_hits += 1
                        continue

                details = re.split("^No \d+$", f.read(), flags=re.M)

            res.append(details.pop(0)) if len(details) > num_hits else None
            assert num_hits == len(details)
            details = np.array(details)[sorted(ts)]
            for i, detail in enumerate(details, 1):
                res.append(f"No {i}")
                res.append(detail)
            with open(f"{o_dir}/{c}/pdb70_hits.hhr", "w") as f:
                print(*res, sep="", file=f)

    # a pdb needs to be copied if it is:
    # 1) a part of sampled training chains, or
    # 2) a template of 1, or
    # 3) a template of validation chains.

    chains_templ = sum(templates, [])

    with open(template_release_dates_cache_path) as f:
        templ_cache = json.load(f)

    obsolete = pd.read_csv(obsolete_pdbs_file_path, delim_whitespace=True)
    assert obsolete.iloc[:, 2].is_unique
    obsolete = dict(obsolete.iloc[:, 2:4].to_records(index=False))

    # add template chains that are in template cache but not in obsolete
    more_templ_chains = []
    for i in chains_templ:
        pdb_id = i.split("_")[0].upper()
        chain_id = i.split("_")[1]
        if pdb_id in obsolete:
            more_templ_chains.append(obsolete[pdb_id].lower() + "_" + chain_id)
    chains_templ.extend(more_templ_chains)

    pdbs_train = set([i.split("_")[0] for i in chains_train])
    pdbs_templ = set([i.split("_")[0] for i in chains_templ])

    with open(
        os.path.join(output_dir, os.path.basename(template_release_dates_cache_path)),
        "w",
    ) as f:
        data = templ_cache
        samples = pdbs_templ
        subsampled_data = {key: data[key] for key in samples}
        json.dump(subsampled_data, f)

    with open(train_chain_data_cache_path) as f:
        train_cache = json.load(f)  # keys are pdbid_chainid

    with open(
        os.path.join(output_dir, os.path.basename(train_chain_data_cache_path)), "w"
    ) as f:
        data = train_cache
        samples = chains_train
        subsampled_data = {key: data[key] for key in samples}
        json.dump(subsampled_data, f)

    subprocess.run(["cp", "-rf", val_data_dir, f"{output_dir}/val/"], check=True)
    subprocess.run(["cp", group_json, output_dir])
    subprocess.run(["cp", obsolete_pdbs_file_path, f"{output_dir}/obsolete.dat"])

    # copy structure file of training chains and template chains
    for pdb in tqdm(pdbs_train | pdbs_templ):
        # Copy the corresponding mmcif file
        if not os.path.isfile(f"{input_mmcif_dir}/{pdb}.cif"):
            print(f"file not exists: {input_mmcif_dir}/{pdb}.cif")
            continue
        subprocess.run(
            [
                "cp",
                f"{input_mmcif_dir}/{pdb}.cif",
                f"{output_dir}/mmcif_files",
            ],
            check=True,
        )

    # build .db and .index for training set
    db_script_dir = f"{os.path.dirname(__file__)}/alignment_db_scripts"
    subprocess.run(
        [
            "python3",
            f"{db_script_dir}/create_alignment_db.py",
            f"{output_dir}/alignments",
            f"{output_dir}/alignment_db",
            "sample",
        ]
    )
    subprocess.run(
        [
            "python3",
            f"{db_script_dir}/unify_alignment_db_indices.py",
            f"{output_dir}/alignment_db",
            f"{output_dir}/alignment_db",
        ]
    )
    # duplicate index
    subprocess.run(
        [
            "python3",
            f"{db_script_dir}/add_index.py",
            "--group_json",
            group_json,
            "--input_index",
            f"{output_dir}/alignment_db/super.index",
            "--output_index",
            f"{output_dir}/alignment_db/dup_super.index",
        ]
    )

    # remove chains not in train_chain_data_cache_path
    with open(f"{output_dir}/alignment_db/dup_super.index") as f:
        ind = {k: v for k, v in json.load(f).items() if k in chains_train}
    with open(f"{output_dir}/alignment_db/dup_super.index", "w") as f:
        json.dump(ind, f)


"""
python3 construct_sample_train_data.py \
    --input_mmcif_dir /scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files \
    --input_alignment_dir /scratch/00946/zzhang/data/openfold/ls6-tacc/alignment_openfold \
    --val_data_dir /scratch/00946/zzhang/data/openfold/cameo/mmcif_files \
    --val_alignment_dir /scratch/00946/zzhang/data/openfold/cameo/alignments \
    --output_dir /scratch/09101/whatever/data/openfold_sample \
    --train_chain_data_cache_path /scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/prot_data_cache.json \
    --template_release_dates_cache_path /scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/mmcif_cache.json \
    --group_json /scratch/00946/zzhang/data/openfold/ls6-tacc/alignment_db/prot_groups.json \
    --obsolete_pdbs_file_path /scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/obsolete.dat \
    --n_samples 100 \
    --n_templs 7
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Subsample from the dataset.")
    parser.add_argument(
        "--input_mmcif_dir",
        required=True,
        help="Input directory containing mmcif files",
    )
    parser.add_argument(
        "--input_alignment_dir",
        required=True,
        help="Input directory containing alignment files",
    )
    parser.add_argument(
        "--val_data_dir",
        required=True,
        help="Input directory containing validation mmcif files",
    )
    parser.add_argument(
        "--val_alignment_dir",
        required=True,
        help="Input directory containing validation alignment files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory to save the subsampled dataset",
    )
    parser.add_argument(
        "--train_chain_data_cache_path",
        required=True,
        help="Path to the train_chain_data_cache json file",
    )
    parser.add_argument(
        "--template_release_dates_cache_path",
        required=True,
        help="Path to the template_release_dates_cache json file",
    )
    parser.add_argument(
        "--obsolete_pdbs_file_path",
        required=True,
    )
    parser.add_argument(
        "--group_json",
        required=True,
    )
    parser.add_argument(
        "--n_samples", type=int, default=100, help="Number of samples to subsample"
    )
    parser.add_argument(
        "--n_templs", type=int, default=100, help="Number of samples to subsample"
    )
    args = parser.parse_args()

    subsample(
        args.input_mmcif_dir,
        args.input_alignment_dir,
        args.val_data_dir,
        args.val_alignment_dir,
        args.output_dir,
        args.train_chain_data_cache_path,
        args.template_release_dates_cache_path,
        args.obsolete_pdbs_file_path,
        args.group_json,
        args.n_samples,
        args.n_templs,
    )

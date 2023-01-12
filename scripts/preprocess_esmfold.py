import os

# import json
import sys
import subprocess
import logging

sys.path = ["/work/09101/whatever/openfold"] + sys.path

import numpy as np
import requests

# from flatten_dict import flatten, unflatten
# from ml_collections import ConfigDict
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from openfold.data.simple_modules import AtlasSimpleSingleDataset

# # fmt: off
# MASK37_ARR = np.array(
#     [
#         [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
#     ]
# )
# # fmt: on

# RES_ARR = "GACSTVIMDNHLWFYPEQRK"
# RES_TO_MASK37 = {res: a for res, a in zip(RES_ARR, MASK37_ARR)}


# def update_config(old: dict, new: dict) -> dict:
#     return unflatten({**flatten(old), **flatten(new)})


def preprocess_pdb(base_dir: str, collection: str, protein_id: str) -> None:
    try:
        res, internal = AtlasSimpleSingleDataset.make_decoy_pdb_compact(
            f"{base_dir}/raw/{collection}/{protein_id}.pdb", return_internal=True
        )
    except (AssertionError, ValueError) as e:
        logging.warning(f"Error {e} for {collection}/{protein_id}.pdb")
        return

    np.savez_compressed(f"{base_dir}/compact/{collection}/{protein_id}.npz", **res)
    res = AtlasSimpleSingleDataset.make_decoy_compact_openfold(
        f"{base_dir}/compact/{collection}/{protein_id}.npz", confidence_threshold=0.5
    )

    for k in res:
        if k in internal:
            assert internal[k].dtype == res[k].dtype
            assert internal[k].shape == res[k].shape
            if k == "all_atom_mask":
                pass
            elif k == "all_atom_positions":
                pass
                # progress_bar.set_postfix(
                #     {"acc loss": f"{np.abs(internal[k] - res[k]).max():.3f}"}
                # )
            else:
                # if not (internal[k] == res[k]).all():
                #     import pdb; pdb.set_trace()
                # assert (internal[k] == res[k]).all(), [
                #     k,
                #     internal[k],
                #     res[k],
                # ]
                if not (internal[k] == res[k]).all():
                    logging.warning(
                        f"Conversion back to OpenFold is incorrect for {base_dir}/{protein_id}.pdb",
                        k,
                        internal[k],
                        res[k],
                    )
                    break


def preprocess_collection(base_dir: str, collection: str) -> None:
    protein_ids = [
        i.split(".")[0]
        for i in os.listdir(f"{base_dir}/raw/{collection}")
        if i.endswith(".pdb")
    ]

    # if parallelize:
    #     Parallel(n_jobs=32)(
    #         delayed(preprocess_pdb)(f"{base_dir}/raw/{collection}", i) for i in tqdm(protein_ids)
    #     )
    # else:
    for protein_id in tqdm(protein_ids):
        preprocess_pdb(f"{base_dir}", collection, protein_id)


def preprocess_chunk(base_dir: str, url: str) -> None:
    assert url.endswith(".tar.gz")
    file_name = os.path.split(url[:-7])[1]
    os.makedirs(f"{base_dir}/{file_name}", exist_ok=True)
    os.makedirs(f"{base_dir}/{file_name}/raw", exist_ok=True)
    os.makedirs(f"{base_dir}/{file_name}/compact", exist_ok=True)
    # subprocess.run(["wget", url], cwd=f"{base_dir}/{file_name}/raw")
    # subprocess.run(
    #     ["tar", "--use-compress-program=pigz", "-xzf", f"{file_name}.tar.gz"],
    #     cwd=f"{base_dir}/{file_name}/raw",
    # )
    # now we have one .tar.gz and a bunch of collections in the raw folder
    collections = [
        i
        for i in os.listdir(f"{base_dir}/{file_name}/raw")
        if os.path.isdir(f"{base_dir}/{file_name}/raw/{i}")
    ]

    if parallelize:
        Parallel(n_jobs=32)(
            delayed(preprocess_collection)(f"{base_dir}/{file_name}", i)
            for i in tqdm(collections)
        )
    else:
        for collection in tqdm(collections):
            preprocess_collection(f"{base_dir}/{file_name}", collection)

    subprocess.run(["rm", "-rf", f"{base_dir}/{file_name}/raw"])


def preprocess_collection_simple(base_dir: str, collection: str, save_dir: str) -> None:
    protein_ids = [
        i.split(".")[0]
        for i in os.listdir(f"{base_dir}/{collection}")
        if i.endswith(".pdb")
    ]

    # if parallelize:
    #     Parallel(n_jobs=32)(
    #         delayed(preprocess_pdb)(f"{base_dir}/raw/{collection}", i) for i in tqdm(protein_ids)
    #     )
    # else:

    os.makedirs(f"{save_dir}/{collection}", exist_ok=True)

    for protein_id in tqdm(protein_ids):
        preprocess_pdb_simple(base_dir, collection, protein_id, save_dir)


def preprocess_pdb_simple(
    base_dir: str, collection: str, protein_id: str, save_dir: str
) -> None:
    try:
        res, internal = AtlasSimpleSingleDataset.make_decoy_pdb_compact(
            f"{base_dir}/{collection}/{protein_id}.pdb", return_internal=True
        )
    except (AssertionError, ValueError) as e:
        logging.warning(f"Error {e} for {collection}/{protein_id}.pdb")
        return

    np.savez_compressed(f"{save_dir}/{collection}/{protein_id}.npz", **res)
    res = AtlasSimpleSingleDataset.make_decoy_compact_openfold(
        f"{save_dir}/{collection}/{protein_id}.npz", confidence_threshold=0.5
    )

    for k in res:
        if k in internal:
            assert internal[k].dtype == res[k].dtype
            assert internal[k].shape == res[k].shape
            if k == "all_atom_mask":
                pass
            elif k == "all_atom_positions":
                pass
                # progress_bar.set_postfix(
                #     {"acc loss": f"{np.abs(internal[k] - res[k]).max():.3f}"}
                # )
            else:
                # if not (internal[k] == res[k]).all():
                #     import pdb; pdb.set_trace()
                # assert (internal[k] == res[k]).all(), [
                #     k,
                #     internal[k],
                #     res[k],
                # ]
                if not (internal[k] == res[k]).all():
                    logging.warning(
                        f"Conversion back to OpenFold is incorrect for {base_dir}/{protein_id}.pdb",
                        k,
                        internal[k],
                        res[k],
                    )
                    break


if __name__ == "__main__":
    log_file = "./preprocess_esmfold_atlas_log.txt"
    logging.basicConfig(filename=log_file)

    # Steps for ESMFold Atlas:
    # - Download one .tar.gz file from the url
    # - Uncompress the file
    # - For all the collections, perform compaction for all pdb files in it. Store the compact format.
    # - Delete the collections and .tar.gz file

    esmfold_atlas_url = "https://raw.githubusercontent.com/facebookresearch/esm/main/scripts/atlas/v0/highquality_clust30/tarballs.txt"
    atlas_name = "esmfold"
    data_dir = "/scratch/09101/whatever/data"
    os.makedirs(f"{data_dir}/{atlas_name}_atlas", exist_ok=True)

    parallelize = True

    # data_dir = "/work/09101/whatever/data/example"
    # collection = "003"

    # with open(f"configs/base.json") as f:
    #     config = json.load(f)
    # with open(f"configs/initial_training.json") as f:
    #     config = update_config(config, json.load(f))
    # with open(f"configs/train.json") as f:
    #     config = update_config(config, json.load(f))

    # urls = [i.strip() for i in requests.get(esmfold_atlas_url).text.splitlines()]
    # if parallelize:
    #     Parallel(n_jobs=1)(
    #         delayed(preprocess_chunk)(f"{data_dir}/{atlas_name}_atlas", i)
    #         for i in tqdm(urls)
    #     )
    # else:
    #     for url in tqdm(urls):
    #         preprocess_chunk(f"{data_dir}/{atlas_name}_atlas", url)

    esmfold_raw_dir = "/scratch/00946/zzhang/data/esmfold"
    save_dir = "/scratch/09101/whatever/data/esmfold_atlas"
    collections = [
        i
        for i in sorted(os.listdir(esmfold_raw_dir))
        if os.path.isdir(f"{esmfold_raw_dir}/{i}")
    ]
    if parallelize:
        n_jobs = 96
        chunk_id = 10
        Parallel(n_jobs=n_jobs)(
            delayed(preprocess_collection_simple)(esmfold_raw_dir, collection, save_dir)
            for collection in tqdm(
                collections[n_jobs * chunk_id : n_jobs * (chunk_id + 1)]
            )
        )
    else:
        for collection in tqdm(collections):
            preprocess_collection_simple(esmfold_raw_dir, collection, save_dir)

    # res_prot = protein.Protein(
    #     atom_positions=res["all_atom_positions"],
    #     aatype=res["aatype"].argmax(axis=1),
    #     atom_mask=res["all_atom_mask"],
    #     residue_index=res["residue_index"]+1,
    #     b_factors=np.zeros_like(res["all_atom_mask"]),  # dummy
    # )
    # with open(f"{data_dir}/{atlas_name}_res/{collection}_{protein_id}.pdb", "w") as f:
    #     print(protein.to_pdb(res_prot), file=f)

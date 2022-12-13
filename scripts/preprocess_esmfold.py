import os
import json
import sys
import subprocess

sys.path = ["."] + sys.path

import numpy as np
import requests
from flatten_dict import flatten, unflatten
from ml_collections import ConfigDict
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


def update_config(old: dict, new: dict) -> dict:
    return unflatten({**flatten(old), **flatten(new)})


if __name__ == "__main__":
    # Steps for ESMFold Atlas:
    # - Download one .tar.gz file from the url
    # - Uncompress the file
    # - For all the collections, perform compaction for all pdb files in it. Store the compact format.
    # - Delete the collections and .tar.gz file

    esmfold_atlas_url = "https://raw.githubusercontent.com/facebookresearch/esm/main/scripts/atlas/v0/highquality_clust30/tarballs.txt"
    atlas_name = "esmfold"
    data_dir = "/scratch/09101/whatever/data"

    # data_dir = "/work/09101/whatever/data/example"
    # collection = "003"

    with open(f"configs/base.json") as f:
        config = json.load(f)
    with open(f"configs/initial_training.json") as f:
        config = update_config(config, json.load(f))
    with open(f"configs/train.json") as f:
        config = update_config(config, json.load(f))

    urls = [i.strip() for i in requests.get(esmfold_atlas_url).text.splitlines()]
    for url in tqdm(urls):
        subprocess.run(["wget", url], cwd=f"{data_dir}/{atlas_name}_atlas/raw")
        subprocess.run(
            ["tar", "-xvzf", os.path.basename(url)],
            cwd=f"{data_dir}/{atlas_name}_atlas/raw",
        )
        # now we have one .tar.gz and a bunch of collections in the raw folder
        collections = [
            i
            for i in os.listdir(f"{data_dir}/{atlas_name}_atlas/raw")
            if os.path.isdir(f"{data_dir}/{atlas_name}_atlas/raw/{i}")
        ]

        for collection in tqdm(collections):
            os.makedirs(
                f"{data_dir}/{atlas_name}_atlas/compact/{collection}", exist_ok=True
            )
            protein_ids = [
                i.split(".")[0]
                for i in os.listdir(f"{data_dir}/{atlas_name}_atlas/raw/{collection}")
                if i.endswith(".pdb")
            ]

            for protein_id in tqdm(protein_ids):
                try:
                    res, internal = AtlasSimpleSingleDataset.make_decoy_pdb_compact(
                        f"{data_dir}/{atlas_name}_atlas/raw/{collection}/{protein_id}.pdb",
                        ConfigDict(config["data"]),
                        return_internal=True,
                    )
                except AssertionError:
                    print(
                        f"Assertion error for {atlas_name}_atlas/raw/{collection}/{protein_id}.pdb"
                    )
                    continue
                np.savez_compressed(
                    f"{data_dir}/{atlas_name}_atlas/compact/{collection}/{protein_id}.npz",
                    **res,
                )
                res = AtlasSimpleSingleDataset.make_decoy_compact_openfold(
                    f"{data_dir}/{atlas_name}_compact/{collection}/{protein_id}.npz",
                    confidence_threshold=0.5,
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
                                print(
                                    f"Conversion back to OpenFold is incorrect for {atlas_name}_atlas/raw/{collection}/{protein_id}.pdb",
                                    k,
                                    internal[k],
                                    res[k],
                                )
                                break

        # delete the collections and .tar.gz file
        subprocess.run(["rm", "-rf", f"{data_dir}/{atlas_name}_atlas/raw/*"])

    # res_prot = protein.Protein(
    #     atom_positions=res["all_atom_positions"],
    #     aatype=res["aatype"].argmax(axis=1),
    #     atom_mask=res["all_atom_mask"],
    #     residue_index=res["residue_index"]+1,
    #     b_factors=np.zeros_like(res["all_atom_mask"]),  # dummy
    # )
    # with open(f"{data_dir}/{atlas_name}_res/{collection}_{protein_id}.pdb", "w") as f:
    #     print(protein.to_pdb(res_prot), file=f)

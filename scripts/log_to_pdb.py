"""
Take prediction (pdb format) of model, evaluate wrt ground truth and save result in csv
"""

import sys
import argparse
import os
from typing import Dict, Union
import json

sys.path = ["/work/09101/whatever/openfold"] + sys.path

import numpy as np
import pandas as pd
import torch

from openfold.np import residue_constants, protein
from openfold.np.protein import Protein, to_pdb
from openfold.data.data_pipeline import _aatype_to_str_sequence
from openfold.data import mmcif_parsing
from openfold.np.residue_constants import restype_atom37_mask as atom_mask_37
from openfold.utils.superimposition import superimpose
from openfold.utils.validation_metrics import (
    drmsd,
    gdt_ts,
    gdt_ha,
)
from openfold.utils.loss import lddt_ca
from pymol import cmd


def log_to_pdb(df_path: str, val_alignment_dir: str, save_dir: str) -> None:
    validation_chains = [
        i
        for i in sorted(os.listdir(val_alignment_dir))
        if os.path.isdir(f"{val_alignment_dir}/{i}")
    ]
    seqs = []
    for c in validation_chains:
        with open(f"{val_alignment_dir}/{c}/bfd_uniclust_hits.a3m") as f:
            assert next(f) == ">query\n"
            seq = next(f)
        seqs.append(seq.strip())
    seq2id = {seq: c for seq, c in zip(seqs, validation_chains)}
    df = pd.read_pickle(df_path)
    for log in df.itertuples():
        aatype = log.sequence
        atom_positions = log.final_atom_positions
        atom_mask = atom_mask_37[aatype]
        b_factors = log.plddt[:, None] * atom_mask
        residue_index = np.arange(len(aatype)) + 1
        seq = _aatype_to_str_sequence(aatype)
        chain_id = seq2id[seq]
        lddt = log.other_metrics["lddt_ca"]

        with open(f"{save_dir}/{chain_id}_{int(lddt * 100)}.pdb", "w") as f:
            print(
                to_pdb(
                    Protein(atom_positions, aatype, atom_mask, residue_index, b_factors)
                ),
                file=f,
            )


def cal_sec_struc_pymol(structure_path: str, chain: str):
    # Close PyMOL
    cmd.reinitialize()

    # Load the mmCIF file into PyMOL
    cmd.load(structure_path, "prot")

    cmd.remove(f"not (chain {chain})")

    # Select all alpha helices in the protein structure
    cmd.select("alpha_helices", "ss h")

    # Calculate the number of alpha helices in the protein
    num_alpha_helices = cmd.count_atoms("alpha_helices")

    # Select all beta sheets in the protein structure
    cmd.select("beta_sheets", "ss s")

    # Calculate the number of beta sheets in the protein
    num_beta_sheets = cmd.count_atoms("beta_sheets")

    # Calculate the total number of amino acids in the protein
    num_amino_acids = cmd.count_atoms("polymer")

    # Calculate the percentage of amino acids that are alpha helices
    percent_alpha_helices = (num_alpha_helices / num_amino_acids) * 100

    # Calculate the percentage of amino acids that are beta sheets
    percent_beta_sheets = (num_beta_sheets / num_amino_acids) * 100
    return percent_alpha_helices, percent_beta_sheets


def evaluate_one(
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    all_atom_mask: np.ndarray,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    eps = 1e-4 for low_prec, otherwise eps = 1e-8
    """
    assert pred_coords.shape[:2] == gt_coords.shape[:2] == all_atom_mask.shape, (
        pred_coords.shape[:2],
        gt_coords.shape[:2],
        all_atom_mask.shape,
    )

    metrics = {}
    # gt_coords = batch["all_atom_positions"]
    # pred_coords = outputs["final_atom_positions"]
    # all_atom_mask = batch["all_atom_mask"]
    to_torch = lambda x: torch.from_numpy(x).unsqueeze(0)

    pred_coords = to_torch(pred_coords)
    gt_coords = to_torch(gt_coords)
    all_atom_mask = to_torch(all_atom_mask)

    # This is super janky for superimposition. Fix later
    gt_coords_masked = gt_coords * all_atom_mask[..., None]
    pred_coords_masked = pred_coords * all_atom_mask[..., None]
    ca_pos = residue_constants.atom_order["CA"]
    gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
    pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
    all_atom_mask_ca = all_atom_mask[..., ca_pos]

    lddt_ca_score = lddt_ca(
        pred_coords,
        gt_coords,
        all_atom_mask,
        eps=eps,
        per_residue=False,
    )

    metrics["lddt_ca"] = lddt_ca_score

    drmsd_ca_score = drmsd(
        pred_coords_masked_ca,
        gt_coords_masked_ca,
        mask=all_atom_mask_ca,  # still required here to compute n
    )

    metrics["drmsd_ca"] = drmsd_ca_score

    superimposed_pred, alignment_rmsd = superimpose(
        gt_coords_masked_ca,
        pred_coords_masked_ca,
        all_atom_mask_ca,
    )
    gdt_ts_score = gdt_ts(superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca)
    gdt_ha_score = gdt_ha(superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca)

    metrics["alignment_rmsd"] = alignment_rmsd
    metrics["gdt_ts"] = gdt_ts_score
    metrics["gdt_ha"] = gdt_ha_score

    return {k: v.item() for k, v in metrics.items()}


def evaluate(
    predicted_dir: str,
    ground_truth_dir: str,
    other_prediction_path: str = None,
    casp15_meta_path: str = None,
) -> pd.DataFrame:
    """For all pdb files in predicted dir, find ground truth structure (mmcif or pdb)
    in ground truth dir (skip if there is no corresponding ground truth), calculate
    these metrics:
    - GDT_TS
    - GDT_HA
    - lDDT_Cα
    - dRMSD_Cα
    - Alignment RMSD

    CASP15 is different in that each protein is evaluated on domains, and one protein can
    actually gives rise to several evaluation result.

    RGN2 does not predict last two residues. So take special care of it.

    Results are saved in a csv file.
    """
    predicteds = sorted(
        i[:-4]
        for i in os.listdir(predicted_dir)
        if i.endswith(".pdb") and os.path.isfile(f"{predicted_dir}/{i}")
    )

    if casp15_meta_path is not None:
        # gt_dir = f"{ground_truth_dir}/whole"
        with open(casp15_meta_path) as f:
            casp15_meta = json.load(f)

    if other_prediction_path is not None:
        other_data_df = pd.read_pickle(other_prediction_path).set_index("name")
    else:
        other_data_df = None

    res = []
    for i in predicteds:

        # get all "parts" for evaluation
        if casp15_meta_path is None:
            gts_for_i = [i]
        else:
            gts_for_i = [
                f"domains/{j[0]}" if j[1] else f"whole/{j[0]}"
                for j in casp15_meta[i]["domains"]
            ]

        with open(f"{predicted_dir}/{i}.pdb") as f:
            predicted = protein.from_pdb_string(f.read())
        pred_seq_str = _aatype_to_str_sequence(predicted.aatype)
        pred_coords = predicted.atom_positions
        if predicted.residue_index[0] == 0:
            residue_index = predicted.residue_index + 1
        else:
            residue_index = predicted.residue_index
        for idx, j in enumerate(gts_for_i):  # iterate through all ground truth
            print(j)
            if os.path.isfile(f"{ground_truth_dir}/{j}.pdb"):
                # ground truth is in pdb format and there is no chain specified.
                file_type = "pdb"
                path = f"{ground_truth_dir}/{j}.pdb"
                with open(path) as f_gt:
                    gt = protein.from_pdb_string(f_gt.read())
                gt_seq_str = _aatype_to_str_sequence(gt.aatype)
                gt_coords = gt.atom_positions
                all_atom_mask = gt.atom_mask
                residue_index_gt = gt.residue_index
                chain = gt.chain_index
            elif os.path.isfile(f"{ground_truth_dir}/{j.split('_')[0]}.pdb"):
                # ground truth is in pdb format and chain is specified.
                file_type = "pdb"
                path = f"{ground_truth_dir}/{j.split('_')[0]}.pdb"
                chain = j.split("_")[1].split(".")[0]
                with open(path) as f_gt:
                    gt = protein.from_pdb_string(f_gt.read(), chain_id=chain)
                gt_seq_str = _aatype_to_str_sequence(gt.aatype)
                gt_coords = gt.atom_positions
                all_atom_mask = gt.atom_mask
                residue_index_gt = gt.residue_index
                chain = gt.chain_index
            elif os.path.isfile(f"{ground_truth_dir}/{j}.cif"):
                # ground truth is in mmcif format and there is no chain specified.
                file_type = "cif"
                path = f"{ground_truth_dir}/{j}.cif"
                with open(path) as f_gt:
                    gt = mmcif_parsing.parse(
                        file_id=j, mmcif_string=f_gt.read()
                    ).mmcif_object
                chain = next(gt.structure.get_chains()).id
                gt_seq_str = gt.chain_to_seqres[chain]
                gt_coords, all_atom_mask = mmcif_parsing.get_atom_coords(
                    mmcif_object=gt,
                    chain_id=chain,
                )
                residue_index_gt = np.arange(gt_coords.shape[0]) + 1
            elif os.path.isfile(f"{ground_truth_dir}/{j.split('_')[0]}.cif"):
                # ground truth is in mmcif format and chain is specified.
                file_type = "cif"
                path = f"{ground_truth_dir}/{j.split('_')[0]}.cif"
                with open(path) as f_gt:
                    gt = mmcif_parsing.parse(
                        file_id=j, mmcif_string=f_gt.read()
                    ).mmcif_object
                chain = j.split("_")[1].split(".")[0]
                gt_seq_str = gt.chain_to_seqres[chain]
                gt_coords, all_atom_mask = mmcif_parsing.get_atom_coords(
                    mmcif_object=gt, chain_id=chain
                )
                residue_index_gt = np.arange(gt_coords.shape[0]) + 1
            else:
                # ground truth is not available
                print(f"No ground truth found for {j}")
                continue

            # QUESTION: How do we skip residues in mmcif file?
            shared_index = np.intersect1d(residue_index, residue_index_gt)
            if casp15_meta_path is not None:
                domains = casp15_meta[i]["domains"][idx][4]
                if domains == [None]:
                    official_range = np.arange(casp15_meta[i]["length"]) + 1
                else:
                    official_range = np.concatenate(
                        [
                            np.arange(i[0], i[1] + 1)
                            for i in casp15_meta[i]["domains"][idx][4]
                        ]
                    )
                # if j == "domains/T1162-D1":
                #     mask = np.ones(168, dtype=bool)
                #     mask[25:30] = 0
                #     t1130 = Protein(
                #         atom_positions=gt_coords[mask],
                #         aatype=gt.aatype[mask],
                #         atom_mask=all_atom_mask[mask],
                #         residue_index=gt.residue_index[mask],
                #         b_factors=gt.b_factors[mask],
                #     )
                #     with open(
                #         "/scratch/09101/whatever/data/casp15/domains/T1162-D1.pdb", "w"
                #     ) as f:
                #         print(protein.to_pdb(t1130), file=f, end="")
                missing = set(residue_index_gt) - set(official_range)
                assert not missing, missing
                shared_index = np.intersect1d(official_range, shared_index)
            shared_index -= 1

            result = {
                "name": i,
                "domain": j.split("/")[1],
                "sequence": "".join(np.array(list(pred_seq_str))[shared_index]),
                "sequence_full": pred_seq_str,
            }

            assert result["sequence"] == "".join(np.array(list(pred_seq_str))[shared_index])
            if not result["sequence"] == gt_seq_str.replace("X", "M"):
                import difflib
                d = difflib.Differ()
                diff = d.compare([result["sequence"]], [gt_seq_str])
                print("\n".join(diff))

            result.update(
                evaluate_one(
                    pred_coords[shared_index],
                    gt_coords,
                    all_atom_mask,
                )
            )

            alpha, beta = cal_sec_struc_pymol(path, chain)
            result["a"] = alpha
            result["b"] = beta

            if casp15_meta_path is not None:
                if other_data_df is None:
                    mean_plddt = None
                else:
                    mean_plddt = other_data_df.loc[i].plddt[shared_index].mean()
                result["category"] = casp15_meta[i]["domains"][idx][1]
                result["casp15_metric"] = casp15_meta[i]["domains"][idx][3]
                result["residue_index"] = shared_index
                result["domains"] = domains
                result["mean_plddt"] = mean_plddt
                result["length"] = len(official_range)
                result["length_full"] = casp15_meta[i]["length"]

            res.append(result)

    ret = pd.DataFrame(res)
    return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val_set",
        help="the validation set to use",
        type=str,
        required=True,
        choices=["cameo", "casp15"],
    )
    parser.add_argument(
        "--result_dir",
        help="the directory that contains the prediction results",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--prediction_path",
        help="dataframe file containing plddt prediction",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    val_set = args.val_set
    result_dir = args.result_dir
    prediction_path = args.prediction_path
    model = args.model

    if val_set == "cameo":
        val_alignment_dir = "/scratch/00946/zzhang/data/openfold/cameo/alignments"
        df = evaluate(
            f"./val_res/cameo_pdb/{model}",
            "/scratch/00946/zzhang/data/openfold/cameo/mmcif_files",
        )
        df.to_csv(f"./val_res/cameo/{model}_metrics.csv", index=False)
    elif val_set == "casp15":
        df = evaluate(
            result_dir,
            f"/scratch/09101/whatever/data/casp15",
            other_prediction_path=prediction_path,
            casp15_meta_path="/scratch/09101/whatever/data/casp15/casp15_regular.json",
        )
        df.to_csv(f"./val_res/casp15/{model}_metrics.csv", index=False)

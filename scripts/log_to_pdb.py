import sys
import argparse
import os

sys.path = ["/work/09101/whatever/openfold"] + sys.path

import numpy as np
import pandas as pd

from openfold.np.protein import Protein, to_pdb
from openfold.data.data_pipeline import _aatype_to_str_sequence
from openfold.np.residue_constants import restype_atom37_mask as atom_mask_37


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
    args = parser.parse_args()

    val_set = args.val_set

    if val_set == "cameo":
        val_alignment_dir = (
            "/scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/val_set/alignments"
        )
        model = "of_baseline_53"
        log_to_pdb(f"./val_res/{model}.pkl", val_alignment_dir, f"./val_res_pdb/{model}")
    elif val_set == "casp15":
        model = "of_baseline_53"
        log_to_pdb(f"./val_casp15_res/{model}.pkl", f"./val_casp15_res_pdb/{model}")

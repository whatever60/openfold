import json
import os
import sys
from typing import Dict, Tuple, List
from ast import literal_eval

# from cdifflib import CSequenceMatcher as SequenceMatcher
import tempfile
import subprocess

sys.path = ["/work/09101/whatever/openfold"] + sys.path

import numpy as np
import pandas as pd
from pymol import cmd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns
from tqdm.auto import tqdm

from openfold.data import templates, parsers, mmcif_parsing
from openfold.data.data_pipeline import _aatype_to_str_sequence


def cal_sec_struc_content(
    val_set_dir, val_alignment_dir
) -> Tuple[List[float], List[float]]:
    """
    From ChatGPT
    """
    id_chains = [
        i.split("_")
        for i in sorted(os.listdir(val_alignment_dir))
        if os.path.isdir(f"{val_alignment_dir}/{i}")
    ]
    # pymol.finish_launching()

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

    percents_a, percents_b = zip(
        *[
            cal_sec_struc_pymol(f"{val_set_dir}/{id_}.cif", chain)
            for id_, chain in id_chains
        ]
    )

    return np.array(percents_a), np.array(percents_b)


def hamming_distance(s1, s2):
    """Calculate the Hamming distance between two strings
    From ChatGPT"""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def similarity_matrix(sequences):
    """Calculate the similarity matrix for a list of sequences using the Hamming distance
    From ChatGPT"""
    n = len(sequences)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = hamming_distance(sequences[i], sequences[j])
            matrix[j, i] = matrix[i, j]
    return matrix


def calc_neff(alignment_dir: str) -> float:
    """Calculate effective MSA depth for all proteins in the alignment dir with HMMBuild.
    Do special processing for problematic alignments.
    Only performed for CAEMO validation set.
    """
    with open(f"{alignment_dir}/bfd_uniclust_hits.a3m") as f1, open(
        f"{alignment_dir}/mgnify_hits.a3m"
    ) as f2, open(f"{alignment_dir}/uniref90_hits.a3m") as f3:
        msa = f1.read()
        msa += "".join(f2.readlines()[2:])
        if alignment_dir.split("/")[-1] == "6p0a_A":
            # remove one bad sequence
            f = f3.readlines()
            f = f[2:8067] + f[8069:]
            msa += "".join(f)
        else:
            msa += "".join(f3.readlines()[2:])

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as msafile:
        msafile.write(msa)
        msafile.flush()
        res = subprocess.run(
            [
                "hmmbuild",
                "--informat",
                "a2m",
                "--seed",
                "42",
                "--amino",
                "/dev/null",
                msafile.name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        neff = res.stdout.splitlines()[-3].split()[5]
    return float(neff)


def count_msa_depth(alignment_dir) -> Tuple[float, int, str]:
    msa_data = {}
    for f in os.listdir(alignment_dir):
        path = os.path.join(alignment_dir, f)
        ext = os.path.splitext(f)[-1]

        if ext == ".a3m":
            with open(path, "r") as fp:
                msa, deletion_matrix = parsers.parse_a3m(fp.read())
            data = {"msa": msa, "deletion_matrix": deletion_matrix}
        elif ext == ".sto":
            with open(path, "r") as fp:
                msa, deletion_matrix, _ = parsers.parse_stockholm(fp.read())
            data = {"msa": msa, "deletion_matrix": deletion_matrix}
        else:
            continue

        msa_data[f] = data
    msas, dms = zip(*[(v["msa"], v["deletion_matrix"]) for v in msa_data.values()])

    # deduplicate deletion_matrices based on msas

    # a = np.concatenate([np.array([list(i) for i in msa]) for msa in msas], axis=0)
    # b = np.concatenate([np.array([list(i) for i in dm]) for dm in dms], axis=0)
    a = np.concatenate([np.array(msa) for msa in msas], axis=0)  # [num_msas, num_res]
    b = np.concatenate([np.array(dm) for dm in dms], axis=0)  # [num_msas, num_res]
    undup_ind = np.sort(np.unique(a, return_index=True)[1])
    msas = a[undup_ind]
    deletion_matrices = b[undup_ind]

    # now we have two numpy arrays.
    # make a similarity matrix (use a random implementation for measuring similarity for now) of the sequences in the MSA profile

    # The following code to compute similarity matrix takes too long.
    # weight = np.array(
    #     [SequenceMatcher(None, i, j).ratio() for i in tqdm(msas) for j in msas]
    # )

    # neff = (1 / weight).sum()
    # mean_neff = (np.array([list(i) for i in msas]) != "-").sum(axis=0).mean() * neff
    mean_neff = len(msas)
    return mean_neff, len(msas), msas[0]


def plot_plddt_lddt(
    val_alignment_dir: str,
    val_set_dir: str,
    log_path: str,
    seq_sim_path: str,
    str_sim_path: str,
    cache_path: str,
    fig_dir: str,
) -> None:
    """
    `log_path` is a path to a pickled pandas dataframe file. It stores metrics of validation set.
    Its columns are:
    - `sequence`: list of amino acid ids
    - `all_atom_positions`: numpy array with shape (num_res, 37, 3). Ground truth protein atom coordinates.
    - `final_atom_positions`: numpy array with shape (num_res, 37, 3). Predicted protein atom coordinates.
    - `plddt`: list of per-amino acid plddt.
    - `tm`: list of per-amino acid ptm.
    - `pae`: numpy arry with shape (num_res, num_res). PAE.
    - `loss`: dict with keys dict_keys(['distogram', 'experimentally_resolved', 'fape', 'plddt_loss', 'masked_msa', 'supervised_chi', 'violation', 'unscaled_loss', 'loss']). Loss components.
    - `other_metrics`: dict with keys keys dict_keys(['lddt_ca', 'drmsd_ca', 'alignment_rmsd', 'gdt_ts', 'gdt_ha']).
    """
    # load model log dataframe
    df_log = pd.read_pickle(log_path)

    df_log["seq"] = df_log.sequence.map(_aatype_to_str_sequence)

    # count similar sequences
    nums = []
    df_seq = pd.read_csv(seq_sim_path, header=None, sep="\t")
    for i in range(df_log.shape[0]):
        row = df_seq[df_seq[1] == f"cameo_{i}"]
        assert row.shape[0] == 1
        rep = row[0].to_list()[0]
        nums.append((df_seq[0] == rep).sum())
    df_log["log_num_similar_sequences"] = np.log10(np.array(nums) + 1)

    # calculate additional properties of proteins in validation set
    protein_ids = [
        i
        for i in sorted(os.listdir(val_alignment_dir))
        if os.path.isdir(f"{val_alignment_dir}/{i}")
    ]
    print(protein_ids)

    # secondary structure components
    percents_a, percents_b = cal_sec_struc_content(val_set_dir, val_alignment_dir)

    # get protein sequence in string format and number of msa sequences
    res = [
        count_msa_depth(f"{val_alignment_dir}/{protein_id}")
        for protein_id in tqdm(protein_ids)
    ]
    _, num_msas, seqs = map(np.array, zip(*res))

    # N_effective
    neffs = [
        calc_neff(f"{val_alignment_dir}/{protein_id}")
        for protein_id in tqdm(protein_ids)
    ]

    # structure similarity
    df_str = pd.read_csv(str_sim_path, header=None, sep="\t")
    with open(cache_path) as f:
        cache = json.load(f).keys()
    df_str = df_str[df_str[1].map(lambda x: x.replace(".cif", "")).isin(cache)]
    nums = []
    query = df_str[1].to_list()
    for name in protein_ids:
        name = name.replace("_", ".cif_")
        nums.append((df_str[1] == name).sum()) if name in query else nums.append(0)

    # add these protein properties to the log dataframe
    df_addi = pd.DataFrame(
        dict(
            name=protein_ids,
            seq=seqs,
            num_msa=num_msas,
            log_num_msa=np.log10(num_msas + 1),
            Neff=neffs,
            log_Neff=np.log1p(neffs),
            percent_α_helix=percents_a,
            percent_β_sheet=percents_b,
            log_num_similar_structures=np.log10(np.array(nums) + 1),
        )
    )
    df_addi = df_addi.sort_values("seq")

    # sort based on amino acid sequence
    df_log = df_log.sort_values("seq")
    assert (df_log.seq.to_numpy() == df_addi.seq.to_numpy()).all()
    df_log = df_log.drop(columns=["seq"]).join(df_addi)

    # add some additional columns
    df_log["length"] = df_log.seq.map(len)
    lddt_ca = df_log.other_metrics.map(lambda x: x["lddt_ca"])
    drmsd_ca = df_log.other_metrics.map(lambda x: x["drmsd_ca"])
    df_log["mean_lddt"] = lddt_ca.map(np.mean)
    df_log["mean_drmsd_ca"] = drmsd_ca.map(np.mean)
    df_log["mean_plddt"] = df_log["plddt"].map(np.mean)
    df_log["violation_loss"] = df_log.loss.map(lambda x: x["violation"])

    model_name = log_path.split("/")[-1].split(".")[0]

    # lDDT_Cα vs. violation loss
    sns.jointplot(data=df_log, x="violation_loss", y="mean_lddt", kind="reg")
    plt.savefig(f"{fig_dir}/{model_name}_lddt_violation.jpg")
    plt.clf()

    # lDDT_Cα vs. log(Neff)
    sns.jointplot(data=df_log, x="log_Neff", y="mean_lddt", kind="reg")
    plt.savefig(f"{fig_dir}/{model_name}_lddt_logneff.jpg")
    plt.clf()

    # lDDT_Cα vs. Neff
    sns.jointplot(data=df_log, x="Neff", y="mean_lddt", kind="reg")
    plt.savefig(f"{fig_dir}/{model_name}_lddt_neff.jpg")
    plt.clf()

    # lDDT_Cα vs. log(num_msa)
    sns.jointplot(data=df_log, x="log_num_msa", y="mean_lddt", kind="reg")
    plt.savefig(f"{fig_dir}/{model_name}_lddt_logmsa.jpg")
    plt.clf()

    # lDDT_Cα vs. num_msa
    sns.jointplot(data=df_log, x="num_msa", y="mean_lddt", kind="reg")
    plt.savefig(f"{fig_dir}/{model_name}_lddt_msa.jpg")
    plt.clf()

    # lDDT_Cα vs. protein length
    sns.jointplot(data=df_log, x="length", y="mean_lddt", kind="reg")
    plt.savefig(f"{fig_dir}/{model_name}_lddt_len.jpg")
    plt.clf()

    # lDDT_Cα vs. plddt
    sns.jointplot(data=df_log, x="mean_plddt", y="mean_lddt", kind="reg")
    plt.savefig(f"{fig_dir}/{model_name}_lddt_plddt.jpg")
    plt.clf()

    # lDDT_Cα vs. α helix percentage
    sns.jointplot(data=df_log, x="percent_α_helix", y="mean_lddt", kind="reg")
    plt.savefig(f"{fig_dir}/{model_name}_lddt_a.jpg")
    plt.clf()

    # lDDT_Cα vs. β sheet percentage
    sns.jointplot(data=df_log, x="percent_β_sheet", y="mean_lddt", kind="reg")
    plt.savefig(f"{fig_dir}/{model_name}_lddt_b.jpg")
    plt.clf()

    # lDDT_Cα vs. dRMSD_Cα
    sns.jointplot(data=df_log, x="mean_drmsd_ca", y="mean_lddt", kind="reg")
    plt.savefig(f"{fig_dir}/{model_name}_lddt_drmsd.jpg")
    plt.clf()

    # lDDT_Cα vs. number of similar sequences
    sns.jointplot(data=df_log, x="log_num_similar_sequences", y="mean_lddt", kind="reg")
    plt.savefig(f"{fig_dir}/{model_name}_lddt_simseq.jpg")
    plt.clf()

    # lDDT_Cα vs. number of similar structures
    sns.jointplot(
        data=df_log, x="log_num_similar_structures", y="mean_lddt", kind="reg"
    )
    plt.savefig(f"{fig_dir}/{model_name}_lddt_simstr.jpg")
    plt.clf()


def scatter_plot(x, y, names, hue, x_label, y_label, save_path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.regplot(x=x, y=y, ax=ax, alpha=0, ax=ax)
    sns.scatterplot(x=x, y=y, hue=hue, alpha=0.5, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    text_axs = []
    for idx, name in enumerate(names):
        # Only append selected countries
        xi, yi = x[idx], y[idx]
        text_axs.append(ax.text(xi, yi, name, fontsize=12))
    adjust_text(
        text_axs, expand_points=(3, 3), arrowprops=dict(arrowstyle="-", lw=1), ax=ax
    )
    fig.savefig(save_path, dpi=150)


def analysis_casp15(pred_path: str, metrics_path: str, fig_dir):
    # sequence, loss, plddt, ptm, ... are here
    df_pred = pd.read_csv(pred_path, index_col="name")
    df_pred["plddt"] = df_pred.plddt.apply(literal_eval)
    df_pred["sequence"] = df_pred.sequence.apply(literal_eval)
    df_pred["loss"] = df_pred.loss.apply(literal_eval)

    df_pred["mean_plddt"] = df_pred.plddt.map(np.mean)

    # metrics, domain, residue_index are here
    df_metrics = pd.read_csv(metrics_path)  
    df_metrics["plddt"] = [
        np.array(df_pred.loc[r.name].plddt)[r.residue_index]
        for r in df_metrics.itertuples()
    ]
    df_metrics["length"] = [
        len(df_pred.loc[r.name].sequence) for r in df_metrics.itertuples()
    ]

    df_metrics["mean_plddt"] = df_metrics.plddt.map(np.mean)
    df_metrics["mean_drmsd_ca"] = df_metrics.drmsd_ca.map(np.mean)

    # remove all whole proteins
    df_metrics = df_metrics[df_metrics.category != ""]

    # There are currently missing for CASP15.
    # lDDT_Cα vs. violation loss
    # lDDT_Cα vs. log(Neff)
    # lDDT_Cα vs. Neff
    # lDDT_Cα vs. log(num_msa)
    # lDDT_Cα vs. num_msa
    # lDDT_Cα vs. number of similar sequences
    # lDDT_Cα vs. number of similar structures

    # lDDT_Cα vs. plddt
    scatter_plot(
        df_metrics.mean_plddt,
        df_metrics.lddt_ca,
        df_metrics.domain,
        df_metrics.category,
        "Mean plDDT",
        "lDDT_Cα",
        f"{fig_dir}/lddt_plddt.jpg",
    )
    # lDDT_Cα vs. dRMSD_Cα
    scatter_plot(
        df_metrics.mean_drmsd_ca,
        df_metrics.lddt_ca,
        df_metrics.domain,
        df_metrics.category,
        "Mean dRMSD_Cα",
        "lDDT_Cα",
        f"{fig_dir}/lddt_drmsd.jpg",
    )
    # lDDT_Cα vs. protein length (of full protein)
    scatter_plot(
        df_metrics.length,
        df_metrics.lddt_ca,
        df_metrics.domain,
        df_metrics.category,
        "Length (of full protein)",
        "lDDT_Cα",
        f"{fig_dir}/lddt_length.jpg",
    )
    # lDDT_Cα vs. α helix percentage (of the fragment)
    scatter_plot(
        df_metrics.a,
        df_metrics.lddt_ca,
        df_metrics.domain,
        df_metrics.category,
        "% α helix composition",
        "lDDT_Cα",
        f"{fig_dir}/lddt_a.jpg",
    )
    # lDDT_Cα vs. β sheet percentage
    scatter_plot(
        df_metrics.b,
        df_metrics.lddt_ca,
        df_metrics.domain,
        df_metrics.category,
        "% β sheet composition",
        "lDDT_Cα",
        f"{fig_dir}/lddt_b.jpg",
    )

    # time vs. protein length
    scatter_plot(
        df_pred.sequence.map(len),
        df_pred.time,
        df_pred.name,
        "Length",
        "Time",
        f"{fig_dir}/time_length.jpg",
    )
    # lDDT_Cα
    scatter_plot(
        np.arange(df_metrics.shape[0]),
        df_metrics.lddt_ca,
        df_metrics.domain,
        df_metrics.category,
        "",
        "lDDT_Cα",
        f"{fig_dir}/lddt.jpg",
    )


if __name__ == "__main__":
    from rich.traceback import install

    install()

    val_alignment_dir = "/scratch/00946/zzhang/data/openfold/cameo/alignments"
    val_set_dir = "/scratch/00946/zzhang/data/openfold/cameo/mmcif_files"

    # cd /scratch/09101/whatever/data/similarity/sequence
    # mmseqs easy-cluster cameo_pdb.fa clusterRes tmp --min-seq-id 0.5 -c 0.8 --cov-mode 1
    seq_sim_path = (
        "/scratch/09101/whatever/data/similarity/sequence/clusterRes_cluster.tsv"
    )

    # cd /scratch/09101/whatever/data/similarity/structure
    # foldseek easy-search /scratch/00946/zzhang/data/openfold/cameo/mmcif_files /scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files aln.m8 tmpFolder
    str_sim_path = "/scratch/09101/whatever/data/similarity/structure/aln.m8"

    cache_path = (
        "/scratch/00946/zzhang/data/openfold/ls6-tacc/gustaf/prot_data_cache.json"
    )
    # protein_id = "1c9t_L"
    # count_msa_depth(f"{val_alignment_dir}/{protein_id}")

    log_dir = "/work/09101/whatever/openfold/val_res"
    model = "of_baseline_53.pkl"
    # plot_plddt_lddt(
    #     val_alignment_dir,
    #     val_set_dir,
    #     f"{log_dir}/{model}",
    #     seq_sim_path,
    #     str_sim_path,
    #     cache_path,
    #     log_dir,
    # )

    analysis_casp15()

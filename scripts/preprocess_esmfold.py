import os
import sys

sys.path = ["."] + sys.path

import json
from typing import Dict

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from flatten_dict import flatten, unflatten
from ml_collections import ConfigDict
from tqdm.auto import tqdm

from openfold.np import protein
from openfold.data.data_pipeline import (
    _aatype_to_str_sequence,
    make_sequence_features,
    empty_template_feats,
)
from openfold.data.feature_pipeline import make_data_config, np_to_tensor_dict
from openfold.data.input_pipeline import nonensembled_transform_fns, compose
from openfold.data.simple_modules import empty_msa_feats
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.np import residue_constants as rc
from openfold.np.residue_constants import (
    restype_rigid_group_default_frame as default_frames,
    restype_atom14_to_rigid_group as group_idx,
    restype_atom14_mask as atom_mask,
    restype_atom14_rigid_group_positions as lit_positions,
    restype_atom37_mask as atom_mask_37,
)
from openfold.utils.tensor_utils import batched_gather


data_dir = "/scratch/09101/whatever/data/esmfold"

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


def make_atom14_masks(aatype: np.ndarray) -> Dict[str, np.ndarray]:
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append(
            [(rc.atom_order[name] if name else 0) for name in atom_names]
        )
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in rc.atom_types
            ]
        )

        restype_atom14_mask.append([(1.0 if name else 0.0) for name in atom_names])

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    # restype_atom14_to_atom37 = torch.tensor(
    #     restype_atom14_to_atom37,
    #     dtype=torch.int32,
    #     device=aatype.device,
    # )
    # restype_atom37_to_atom14 = torch.tensor(
    #     restype_atom37_to_atom14,
    #     dtype=torch.int32,
    #     device=aatype.device,
    # )
    # restype_atom14_mask = torch.tensor(
    #     restype_atom14_mask,
    #     dtype=torch.float32,
    #     device=aatype.device,
    # )
    # protein_aatype = aatype.to(torch.long)
    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14)
    restype_atom14_mask = np.array(restype_atom14_mask)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[aatype]
    residx_atom14_mask = restype_atom14_mask[aatype]

    res = {}
    res["atom14_atom_exists"] = residx_atom14_mask
    res["residx_atom14_to_atom37"] = residx_atom14_to_atom37

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[aatype]
    res["residx_atom37_to_atom14"] = residx_atom37_to_atom14

    # create the corresponding mask
    # restype_atom37_mask = torch.zeros(
    #     [21, 37], dtype=torch.float32, device=aatype.device
    # )
    restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[aatype]
    res["atom37_atom_exists"] = residx_atom37_mask

    return res


def make_decoy_pdb_compact(
    pdb_path: str, config: dict, return_internal: bool = False
) -> Dict[str, np.ndarray]:
    with open(pdb_path) as f:
        protein_object = protein.from_pdb_string(f.read())
    protein_id = os.path.splitext(os.path.basename(pdb_path))[0]
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    num_res = len(aatype)
    pdb_feats: dict = make_sequence_features(
        sequence=sequence, description=protein_id, num_res=num_res
    )
    pdb_feats["all_atom_positions"] = protein_object.atom_positions.astype(np.float32)
    # resolution, is_distillation are not needed.
    # confidence mask is not added on all_atom_mask.
    all_atom_mask = protein_object.atom_mask.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask
    pdb_feats.update(empty_template_feats(len(sequence)))
    pdb_feats.update(empty_msa_feats(sequence))

    # high_confidence = protein_object.b_factors > confidence_threshold
    cfg, feature_names = make_data_config(config, mode="train", num_res=num_res)
    tensor_dict = np_to_tensor_dict(np_example=pdb_feats, features=feature_names)
    with torch.no_grad():
        # features = process_tensors_from_config()
        nonensembled = nonensembled_transform_fns(
            common_cfg=cfg["common"], mode_cfg=cfg["train"]
        )
        tensor_dict = compose(nonensembled)(tensor_dict)
    # now let's apply some assertions to make sure that decoy structure is indeed as simple as we thought.
    assert tensor_dict["seq_length"] == num_res
    assert (tensor_dict["residue_index"].numpy() == np.arange(num_res)).all()
    assert (tensor_dict["between_segment_residues"] == 0).all()

    torsion_sin, torsion_cos = (
        tensor_dict["torsion_angles_sin_cos"][:, 2:].permute(2, 0, 1).numpy()
    )
    a_acos = np.arccos(torsion_cos)
    angle = np.degrees(a_acos)
    angle[torsion_sin < 0] = np.degrees(-a_acos)[torsion_sin < 0] % 360
    frame_rot_mat = tensor_dict["backbone_rigid_tensor"][:, :3, :3].numpy()
    frame_rot_euler = (
        R.from_matrix(frame_rot_mat).as_euler("zyx", degrees=True).astype(np.float32)
    )

    # check that accuracy is preserved, by circular consistency.
    # frame_rot_mat_back = R.from_euler("zyx", frame_rot_euler, degrees=True).as_matrix()
    # frame_rot_euler_back = R.from_matrix(frame_rot_mat_back).as_euler(
    #     "zyx", degrees=True
    # )
    # print(np.abs(frame_rot_euler - frame_rot_euler_back).max())

    frame_trans = tensor_dict["backbone_rigid_tensor"][:, :3, 3].numpy()

    confidence = (
        np.ma.masked_array(protein_object.b_factors, mask=1 - all_atom_mask)
        .max(axis=1, keepdims=True)
        .astype(np.float32)
    )
    # 1 + 5 + 3 + 3 = 12.
    res = {
        "aatype": tensor_dict["aatype"].numpy().astype(np.int8),
        "data": np.concatenate(
            [confidence, angle, frame_rot_euler, frame_trans], axis=1
        ),
    }
    return res, pdb_feats if return_internal else res


def atom14_to_atom37(atom14: np.ndarray, aatype: np.ndarray) -> np.ndarray:
    # atom16: [num_res, 14, 3]
    batch: Dict[str, np.ndarray] = make_atom14_masks(aatype)
    atom37_data = batched_gather(
        torch.from_numpy(atom14),
        torch.from_numpy(batch["residx_atom37_to_atom14"]),
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )

    atom37_data = atom37_data * batch["atom37_atom_exists"][..., None]

    return atom37_data.numpy()


def make_decoy_compact_openfold(
    compact_path: str, confidence_threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    structure = np.load(compact_path)
    aatype = structure["aatype"]
    sequence = _aatype_to_str_sequence(aatype)
    data = structure["data"]
    confidence = data[:, 0]
    torsion_angle = data[:, 1:6]
    frame_rot_euler = data[:, 6:9]
    frame_trans = data[:, 9:12]

    torsion_angle = np.deg2rad(torsion_angle)
    # [num_res, 5, 2]
    torsion_sin_cos = np.concatenate(
        [
            np.full((aatype.shape[0], 2, 2), -999),  # dummy
            np.stack([np.sin(torsion_angle), np.cos(torsion_angle)], axis=2),
        ],
        axis=1,
    )
    rot_mats = R.from_euler("zyx", frame_rot_euler, degrees=True).as_matrix()
    backb_to_global = Rigid(
        Rotation(rot_mats=torch.from_numpy(rot_mats), quats=None),
        torch.from_numpy(frame_trans),
    )
    default_frames_ = torch.from_numpy(default_frames)
    group_idx_ = torch.from_numpy(group_idx)
    atom_mask_ = torch.from_numpy(atom_mask)
    lit_positions_ = torch.from_numpy(lit_positions)
    aatype_ = torch.from_numpy(aatype).long()

    pred_xyz = frames_and_literature_positions_to_atom14_pos(
        torsion_angles_to_frames(
            backb_to_global, torch.from_numpy(torsion_sin_cos), aatype_, default_frames_
        ),
        aatype_,
        default_frames_,
        group_idx_,
        atom_mask_,
        lit_positions_,
    ).numpy()
    # [num_res, 37, 3]
    pred_xyz = atom14_to_atom37(pred_xyz, aatype)
    all_atom_mask = atom_mask_37[aatype]

    seq_len = len(aatype)
    res: dict = make_sequence_features(
        sequence=sequence,
        description=os.path.splitext(os.path.basename(compact_path))[0],
        num_res=seq_len,
    )
    res.update(
        {
            "all_atom_positions": pred_xyz,
            "all_atom_mask": (
                all_atom_mask * (confidence > confidence_threshold)[:, None]
            ).astype(np.float32),
            "resolution": np.array([0.1]),
            "is_distillation": np.array([True]),
        }
    )
    res.update(empty_template_feats(seq_len))
    res.update(empty_msa_feats(sequence))
    return res


def update_config(old: dict, new: dict) -> dict:
    return unflatten({**flatten(old), **flatten(new)})


if __name__ == "__main__":
    data_dir = "/work/09101/whatever/data/example"
    atlas_name = "esmfold"
    collection = "003"
    progress_bar = tqdm(os.listdir(f"{data_dir}/{atlas_name}/{collection}"))
    for protein_id in progress_bar:
        if not protein_id.endswith(".pdb"):
            continue
        protein_id = os.path.splitext(protein_id)[0]
        with open(f"configs/base.json") as f:
            config = json.load(f)
        with open(f"configs/initial_training.json") as f:
            config = update_config(config, json.load(f))
        with open(f"configs/train.json") as f:
            config = update_config(config, json.load(f))

        res, internal = make_decoy_pdb_compact(
            f"{data_dir}/{atlas_name}/{collection}/{protein_id}.pdb",
            ConfigDict(config["data"]),
            return_internal=True,
        )
        np.savez_compressed(
            f"{data_dir}/{atlas_name}_compact/{collection}/{protein_id}.npz", **res
        )
        res = make_decoy_compact_openfold(
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
                    progress_bar.set_postfix(
                        {"acc loss": f"{np.abs(internal[k] - res[k]).max():.3f}"}
                    )
                else:
                    # if not (internal[k] == res[k]).all():
                    #     import pdb; pdb.set_trace()
                    assert (internal[k] == res[k]).all(), [k, internal[k], res[k]]

    # res_prot = protein.Protein(
    #     atom_positions=res["all_atom_positions"],
    #     aatype=res["aatype"].argmax(axis=1),
    #     atom_mask=res["all_atom_mask"],
    #     residue_index=res["residue_index"]+1,
    #     b_factors=np.zeros_like(res["all_atom_mask"]),  # dummy
    # )
    # with open(f"{data_dir}/{atlas_name}_res/{collection}_{protein_id}.pdb", "w") as f:
    #     print(protein.to_pdb(res_prot), file=f)

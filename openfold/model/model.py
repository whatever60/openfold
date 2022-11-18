# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial, reduce
import sys
import math
from operator import mul
import weakref

import torch
import torch.nn as nn

from openfold.model.embedders import (
    InputEmbedder,
    RecyclingEmbedder,
    TemplateAngleEmbedder,
    TemplatePairEmbedder,
    ExtraMSAEmbedder,
)
from openfold.model.evoformer import (
    EvoformerStack,
    ExtraMSAStack,
    ExtraMSABlock,
    EvoformerBlock,
)
from openfold.model.heads import AuxiliaryHeads
from openfold.model.structure_module import StructureModule
from openfold.model.template import (
    TemplatePairStack,
    TemplatePointwiseAttention,
    embed_templates_average,
    embed_templates_offload,
    TemplatePairStackBlock,
)
import openfold.np.residue_constants as residue_constants
from openfold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    build_template_angle_feat,
    build_template_pair_feat,
    atom14_to_atom37,
)
from openfold.utils.loss import (
    compute_plddt,
)
from openfold.utils.tensor_utils import (
    add,
    dict_multimap,
    tensor_tree_map,
)
from openfold.utils.checkpointing import checkpoint_blocks, get_checkpoint_fn
from openfold.utils.rigid_utils import Rotation, Rigid
from openfold.utils.tensor_utils import (
    dict_multimap,
    permute_final_dims,
    flatten_final_dims,
)

attn_core_inplace_cuda = importlib.import_module("attn_core_inplace_cuda")


class AlphaFold(nn.Module):
    """
    Alphafold 2.

    Implements Algorithm 2 (but with training).
    """

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(AlphaFold, self).__init__()

        self.globals = config.globals
        self.config = config.model
        self.template_config = self.config.template
        self.extra_msa_config = self.config.extra_msa

        # Main trunk + structure module
        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"],
        )
        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
        )

        if self.template_config.enabled:
            self.template_angle_embedder = TemplateAngleEmbedder(
                **self.template_config["template_angle_embedder"],
            )
            self.template_pair_embedder = TemplatePairEmbedder(
                **self.template_config["template_pair_embedder"],
            )
            self.template_pair_stack = TemplatePairStack(
                **self.template_config["template_pair_stack"],
            )
            self.template_pointwise_att = TemplatePointwiseAttention(
                **self.template_config["template_pointwise_attention"],
            )

        if self.extra_msa_config.enabled:
            self.extra_msa_embedder = ExtraMSAEmbedder(
                **self.extra_msa_config["extra_msa_embedder"],
            )
            self.extra_msa_stack = ExtraMSAStack(
                **self.extra_msa_config["extra_msa_stack"],
            )

        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )
        self.structure_module = StructureModule(
            **self.config["structure_module"],
        )
        self.aux_heads = AuxiliaryHeads(
            self.config["heads"],
        )

    # def embed_templates(self, batch, z, pair_mask, templ_dim, inplace_safe):
    #     if(self.template_config.offload_templates):
    #         return embed_templates_offload(self,
    #             batch, z, pair_mask, templ_dim, inplace_safe=inplace_safe,
    #         )
    #     elif(self.template_config.average_templates):
    #         return embed_templates_average(self,
    #             batch, z, pair_mask, templ_dim, inplace_safe=inplace_safe,
    #         )

    #     # Embed the templates one at a time (with a poor man's vmap)
    #     pair_embeds = []
    #     n = z.shape[-2]
    #     n_templ = batch["template_aatype"].shape[templ_dim]

    #     if(inplace_safe):
    #         # We'll preallocate the full pair tensor now to avoid manifesting
    #         # a second copy during the stack later on
    #         t_pair = z.new_zeros(
    #             z.shape[:-3] +
    #             (n_templ, n, n, self.globals.c_t)
    #         )

    #     for i in range(n_templ):
    #         idx = batch["template_aatype"].new_tensor(i)
    #         single_template_feats = tensor_tree_map(
    #             lambda t: torch.index_select(t, templ_dim, idx).squeeze(templ_dim),
    #             batch,
    #         )

    #         # [*, N, N, C_t]
    #         t = build_template_pair_feat(
    #             single_template_feats,
    #             use_unit_vector=self.config.template.use_unit_vector,
    #             inf=self.config.template.inf,
    #             eps=self.config.template.eps,
    #             **self.config.template.distogram,
    #         ).to(z.dtype)
    #         t = self.template_pair_embedder(t)

    #         if(inplace_safe):
    #             t_pair[..., i, :, :, :] = t
    #         else:
    #             pair_embeds.append(t)

    #         del t

    #     if(not inplace_safe):
    #         t_pair = torch.stack(pair_embeds, dim=templ_dim)

    #     del pair_embeds

    #     # [*, S_t, N, N, C_z]
    #     t = self.template_pair_stack(
    #         t_pair,
    #         pair_mask.unsqueeze(-3).to(dtype=z.dtype),
    #         chunk_size=self.globals.chunk_size,
    #         use_lma=self.globals.use_lma,
    #         inplace_safe=inplace_safe,
    #         _mask_trans=self.config._mask_trans,
    #     )
    #     del t_pair

    #     # [*, N, N, C_z]
    #     t = self.template_pointwise_att(
    #         t,
    #         z,
    #         template_mask=batch["template_mask"].to(dtype=z.dtype),
    #         use_lma=self.globals.use_lma,
    #     )

    #     t_mask = torch.sum(batch["template_mask"], dim=-1) > 0
    #     # Append singletons
    #     t_mask = t_mask.reshape(
    #         *t_mask.shape, *([1] * (len(t.shape) - len(t_mask.shape)))
    #     )

    #     if(inplace_safe):
    #         t *= t_mask
    #     else:
    #         t = t * t_mask

    #     ret = {}

    #     ret.update({"template_pair_embedding": t})

    #     del t

    #     if self.config.template.embed_angles:
    #         template_angle_feat = build_template_angle_feat(
    #             batch
    #         )

    #         # [*, S_t, N, C_m]
    #         a = self.template_angle_embedder(template_angle_feat)

    #         ret["template_angle_embedding"] = a

    #     return ret

    # def iteration(self, feats, prevs, _recycle=True):
    #     # Primary output dictionary
    #     outputs = {}

    #     # This needs to be done manually for DeepSpeed's sake
    #     dtype = next(self.parameters()).dtype
    #     for k in feats:
    #         if(feats[k].dtype == torch.float32):
    #             feats[k] = feats[k].to(dtype=dtype)

    #     # Grab some data about the input
    #     batch_dims = feats["target_feat"].shape[:-2]
    #     no_batch_dims = len(batch_dims)
    #     n = feats["target_feat"].shape[-2]
    #     n_seq = feats["msa_feat"].shape[-3]
    #     device = feats["target_feat"].device

    #     # Controls whether the model uses in-place operations throughout
    #     # The dual condition accounts for activation checkpoints
    #     inplace_safe = not (self.training or torch.is_grad_enabled())

    #     # Prep some features
    #     seq_mask = feats["seq_mask"]
    #     pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
    #     msa_mask = feats["msa_mask"]

    #     ## Initialize the MSA and pair representations

    #     # m: [*, S_c, N, C_m]
    #     # z: [*, N, N, C_z]
    #     m, z = self.input_embedder(
    #         feats["target_feat"],
    #         feats["residue_index"],
    #         feats["msa_feat"],
    #         inplace_safe=inplace_safe,
    #     )

    #     # Unpack the recycling embeddings. Removing them from the list allows
    #     # them to be freed further down in this function, saving memory
    #     m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)])

    #     # Initialize the recycling embeddings, if needs be
    #     if None in [m_1_prev, z_prev, x_prev]:
    #         # [*, N, C_m]
    #         m_1_prev = m.new_zeros(
    #             (*batch_dims, n, self.config.input_embedder.c_m),
    #             requires_grad=False,
    #         )

    #         # [*, N, N, C_z]
    #         z_prev = z.new_zeros(
    #             (*batch_dims, n, n, self.config.input_embedder.c_z),
    #             requires_grad=False,
    #         )

    #         # [*, N, 3]
    #         x_prev = z.new_zeros(
    #             (*batch_dims, n, residue_constants.atom_type_num, 3),
    #             requires_grad=False,
    #         )

    #     x_prev = pseudo_beta_fn(
    #         feats["aatype"], x_prev, None
    #     ).to(dtype=z.dtype)

    #     # The recycling embedder is memory-intensive, so we offload first
    #     if(self.globals.offload_inference and inplace_safe):
    #         m = m.cpu()
    #         z = z.cpu()

    #     # m_1_prev_emb: [*, N, C_m]
    #     # z_prev_emb: [*, N, N, C_z]
    #     m_1_prev_emb, z_prev_emb = self.recycling_embedder(
    #         m_1_prev,
    #         z_prev,
    #         x_prev,
    #         inplace_safe=inplace_safe,
    #     )

    #     if(self.globals.offload_inference and inplace_safe):
    #         m = m.to(m_1_prev_emb.device)
    #         z = z.to(z_prev.device)

    #     # [*, S_c, N, C_m]
    #     m[..., 0, :, :] += m_1_prev_emb

    #     # [*, N, N, C_z]
    #     z = add(z, z_prev_emb, inplace=inplace_safe)

    #     # Deletions like these become significant for inference with large N,
    #     # where they free unused tensors and remove references to others such
    #     # that they can be offloaded later
    #     del m_1_prev, z_prev, x_prev, m_1_prev_emb, z_prev_emb

    #     # Embed the templates + merge with MSA/pair embeddings
    #     if self.config.template.enabled:
    #         template_feats = {
    #             k: v for k, v in feats.items() if k.startswith("template_")
    #         }
    #         template_embeds = self.embed_templates(
    #             template_feats,
    #             z,
    #             pair_mask.to(dtype=z.dtype),
    #             no_batch_dims,
    #             inplace_safe=inplace_safe,
    #         )

    #         # [*, N, N, C_z]
    #         z = add(z,
    #             template_embeds.pop("template_pair_embedding"),
    #             inplace_safe,
    #         )

    #         if "template_angle_embedding" in template_embeds:
    #             # [*, S = S_c + S_t, N, C_m]
    #             m = torch.cat(
    #                 [m, template_embeds["template_angle_embedding"]],
    #                 dim=-3
    #             )

    #             # [*, S, N]
    #             torsion_angles_mask = feats["template_torsion_angles_mask"]
    #             msa_mask = torch.cat(
    #                 [feats["msa_mask"], torsion_angles_mask[..., 2]],
    #                 dim=-2
    #             )

    #     # Embed extra MSA features + merge with pairwise embeddings
    #     if self.config.extra_msa.enabled:
    #         # [*, S_e, N, C_e]
    #         a = self.extra_msa_embedder(build_extra_msa_feat(feats))

    #         if(self.globals.offload_inference):
    #             # To allow the extra MSA stack (and later the evoformer) to
    #             # offload its inputs, we remove all references to them here
    #             input_tensors = [a, z]
    #             del a, z

    #             # [*, N, N, C_z]
    #             z = self.extra_msa_stack._forward_offload(
    #                 input_tensors,
    #                 msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
    #                 chunk_size=self.globals.chunk_size,
    #                 use_lma=self.globals.use_lma,
    #                 pair_mask=pair_mask.to(dtype=m.dtype),
    #                 _mask_trans=self.config._mask_trans,
    #             )

    #             del input_tensors
    #         else:
    #             # [*, N, N, C_z]
    #             z = self.extra_msa_stack(
    #                 a, z,
    #                 msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
    #                 chunk_size=self.globals.chunk_size,
    #                 use_lma=self.globals.use_lma,
    #                 pair_mask=pair_mask.to(dtype=m.dtype),
    #                 inplace_safe=inplace_safe,
    #                 _mask_trans=self.config._mask_trans,
    #             )

    #     # Run MSA + pair embeddings through the trunk of the network
    #     # m: [*, S, N, C_m]
    #     # z: [*, N, N, C_z]
    #     # s: [*, N, C_s]
    #     if(self.globals.offload_inference):
    #         input_tensors = [m, z]
    #         del m, z
    #         m, z, s = self.evoformer._forward_offload(
    #             input_tensors,
    #             msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
    #             pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
    #             chunk_size=self.globals.chunk_size,
    #             use_lma=self.globals.use_lma,
    #             _mask_trans=self.config._mask_trans,
    #         )

    #         del input_tensors
    #     else:
    #         m, z, s = self.evoformer(
    #             m,
    #             z,
    #             msa_mask=msa_mask.to(dtype=m.dtype),
    #             pair_mask=pair_mask.to(dtype=z.dtype),
    #             chunk_size=self.globals.chunk_size,
    #             use_lma=self.globals.use_lma,
    #             use_flash=self.globals.use_flash,
    #             inplace_safe=inplace_safe,
    #             _mask_trans=self.config._mask_trans,
    #         )

    #     outputs["msa"] = m[..., :n_seq, :, :]
    #     outputs["pair"] = z
    #     outputs["single"] = s

    #     del z

    #     # Predict 3D structure
    #     outputs["sm"] = self.structure_module(
    #         outputs,
    #         feats["aatype"],
    #         mask=feats["seq_mask"].to(dtype=s.dtype),
    #         inplace_safe=inplace_safe,
    #         _offload_inference=self.globals.offload_inference,
    #     )
    #     outputs["final_atom_positions"] = atom14_to_atom37(
    #         outputs["sm"]["positions"][-1], feats
    #     )
    #     outputs["final_atom_mask"] = feats["atom37_atom_exists"]
    #     outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

    #     # Save embeddings for use during the next recycling iteration

    #     # [*, N, C_m]
    #     m_1_prev = m[..., 0, :, :]

    #     # [*, N, N, C_z]
    #     z_prev = outputs["pair"]

    #     # [*, N, 3]
    #     x_prev = outputs["final_atom_positions"]

    #     return outputs, m_1_prev, z_prev, x_prev

    def forward(self, batch):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "msa_feat" ([*, N_seq, N_res, C_msa])
                        MSA features, constructed as in the supplement.
                        C_msa is config.model.input_embedder.msa_dim.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "msa_mask" ([*, N_seq, N_res])
                        MSA mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
                    "extra_msa_mask" ([*, N_extra, N_res])
                        Extra MSA mask
                    "template_mask" ([*, N_templ])
                        Template mask (on the level of templates, not
                        residues)
                    "template_aatype" ([*, N_templ, N_res])
                        Tensor of template residue indices (indices greater
                        than 19 are clamped to 20 (Unknown))
                    "template_all_atom_positions"
                        ([*, N_templ, N_res, 37, 3])
                        Template atom coordinates in atom37 format
                    "template_all_atom_mask" ([*, N_templ, N_res, 37])
                        Template atom coordinate mask
                    "template_pseudo_beta" ([*, N_templ, N_res, 3])
                        Positions of template carbon "pseudo-beta" atoms
                        (i.e. C_beta for all residues but glycine, for
                        for which C_alpha is used instead)
                    "template_pseudo_beta_mask" ([*, N_templ, N_res])
                        Pseudo-beta mask
        """
        # Initialize recycling embeddings
        attns = {}
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]

        is_grad_enabled = torch.is_grad_enabled()

        # Main recycling loop
        num_iters = batch["aatype"].shape[-1]
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # =====================================================================
                # Run the next iteration of the model
                # outputs, m_1_prev, z_prev, x_prev = self.iteration(
                #     feats,
                #     prevs,
                #     _recycle=(num_iters > 1)
                # )

                # Primary output dictionary
                outputs = {}

                # This needs to be done manually for DeepSpeed's sake
                dtype = next(self.parameters()).dtype
                for k in feats:
                    if feats[k].dtype == torch.float32:
                        feats[k] = feats[k].to(dtype=dtype)

                # Grab some data about the input
                batch_dims = feats["target_feat"].shape[:-2]
                no_batch_dims = len(batch_dims)
                n = feats["target_feat"].shape[-2]
                n_seq = feats["msa_feat"].shape[-3]
                device = feats["target_feat"].device

                # Controls whether the model uses in-place operations throughout
                # The dual condition accounts for activation checkpoints
                inplace_safe = not (self.training or torch.is_grad_enabled())

                # Prep some features
                seq_mask = feats["seq_mask"]
                pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
                msa_mask = feats["msa_mask"]

                ## Initialize the MSA and pair representations

                # m: [*, S_c, N, C_m]
                # z: [*, N, N, C_z]
                m, z = self.input_embedder(
                    feats["target_feat"],
                    feats["residue_index"],
                    feats["msa_feat"],
                    inplace_safe=inplace_safe,
                )

                # Unpack the recycling embeddings. Removing them from the list allows
                # them to be freed further down in this function, saving memory
                m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)])

                # Initialize the recycling embeddings, if needs be
                if None in [m_1_prev, z_prev, x_prev]:
                    # [*, N, C_m]
                    m_1_prev = m.new_zeros(
                        (*batch_dims, n, self.config.input_embedder.c_m),
                        requires_grad=False,
                    )

                    # [*, N, N, C_z]
                    z_prev = z.new_zeros(
                        (*batch_dims, n, n, self.config.input_embedder.c_z),
                        requires_grad=False,
                    )

                    # [*, N, 3]
                    x_prev = z.new_zeros(
                        (*batch_dims, n, residue_constants.atom_type_num, 3),
                        requires_grad=False,
                    )

                x_prev = pseudo_beta_fn(feats["aatype"], x_prev, None).to(dtype=z.dtype)

                # The recycling embedder is memory-intensive, so we offload first
                if self.globals.offload_inference and inplace_safe:
                    m = m.cpu()
                    z = z.cpu()

                # m_1_prev_emb: [*, N, C_m]
                # z_prev_emb: [*, N, N, C_z]
                m_1_prev_emb, z_prev_emb = self.recycling_embedder(
                    m_1_prev,
                    z_prev,
                    x_prev,
                    inplace_safe=inplace_safe,
                )

                if self.globals.offload_inference and inplace_safe:
                    m = m.to(m_1_prev_emb.device)
                    z = z.to(z_prev.device)

                # [*, S_c, N, C_m]
                m[..., 0, :, :] += m_1_prev_emb

                # [*, N, N, C_z]
                z = add(z, z_prev_emb, inplace=inplace_safe)

                # Deletions like these become significant for inference with large N,
                # where they free unused tensors and remove references to others such
                # that they can be offloaded later
                del m_1_prev, z_prev, x_prev, m_1_prev_emb, z_prev_emb

                if self.config.template.enabled:
                    # =====================================================================
                    # start of template stack
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # Embed the templates + merge with MSA/pair embeddings
                    template_feats = {
                        k: v for k, v in feats.items() if k.startswith("template_")
                    }
                    # template_embeds = self.embed_templates(
                    #     template_feats,
                    #     z,
                    #     pair_mask.to(dtype=z.dtype),
                    #     no_batch_dims,
                    #     inplace_safe=inplace_safe,
                    # )
                    pair_mask = pair_mask.to(dtype=z.dtype)
                    if self.template_config.offload_templates:
                        return embed_templates_offload(
                            self,
                            template_feats,
                            z,
                            pair_mask,
                            no_batch_dims,
                            inplace_safe=inplace_safe,
                        )
                    elif self.template_config.average_templates:
                        return embed_templates_average(
                            self,
                            template_feats,
                            z,
                            pair_mask,
                            no_batch_dims,
                            inplace_safe=inplace_safe,
                        )

                    # Embed the templates one at a time (with a poor man's vmap)
                    pair_embeds = []
                    n = z.shape[-2]
                    n_templ = template_feats["template_aatype"].shape[no_batch_dims]

                    if inplace_safe:
                        # We'll preallocate the full pair tensor now to avoid manifesting
                        # a second copy during the stack later on
                        t_pair = z.new_zeros(
                            z.shape[:-3] + (n_templ, n, n, self.globals.c_t)
                        )
                    else:
                        t_pair = torch.stack(pair_embeds, dim=no_batch_dims)

                    for i in range(n_templ):
                        idx = template_feats["template_aatype"].new_tensor(i)
                        single_template_feats = tensor_tree_map(
                            lambda t: torch.index_select(t, no_batch_dims, idx).squeeze(
                                no_batch_dims
                            ),
                            template_feats,
                        )

                        # [*, N, N, C_t]
                        t = build_template_pair_feat(
                            single_template_feats,
                            use_unit_vector=self.config.template.use_unit_vector,
                            inf=self.config.template.inf,
                            eps=self.config.template.eps,
                            **self.config.template.distogram,
                        ).to(z.dtype)
                        t = self.template_pair_embedder(t)

                        if inplace_safe:
                            t_pair[..., i, :, :, :] = t
                        else:
                            pair_embeds.append(t)

                        del t

                    # if(not inplace_safe):
                    #     t_pair = torch.stack(pair_embeds, dim=no_batch_dims)

                    del pair_embeds

                    # =====================================================================
                    # start of template pair stack
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # add attention code
                    # [*, S_t, N, N, C_z]
                    # t = self.template_pair_stack(
                    #     t_pair,
                    #     pair_mask.unsqueeze(-3).to(dtype=z.dtype),
                    #     chunk_size=self.globals.chunk_size,
                    #     use_lma=self.globals.use_lma,
                    #     inplace_safe=inplace_safe,
                    #     _mask_trans=self.config._mask_trans,
                    # )
                    templ_mask = pair_mask.unsqueeze(-3).to(dtype=z.dtype)
                    if templ_mask.shape[-3] == 1:
                        expand_idx = list(templ_mask.shape)
                        expand_idx[-3] = t_pair.shape[-4]
                        templ_mask = templ_mask.expand(*expand_idx)

                    blocks = [
                        partial(
                            b,
                            mask=templ_mask,
                            chunk_size=self.globals.chunk_size,
                            use_lma=self.globals.use_lma,
                            inplace_safe=inplace_safe,
                            _mask_trans=self.config._mask_trans,
                        )
                        for b in self.blocks
                    ]

                    if (
                        self.globals.chunk_size is not None
                        and self.template_pair_stack.chunk_size_tuner is not None
                    ):
                        assert not self.training
                        tuned_chunk_size = (
                            self.template_pair_stack.chunk_size_tuner.tune_chunk_size(
                                representative_fn=blocks[0],
                                args=(t_pair.clone(),),
                                min_chunk_size=self.globals.chunk_size,
                            )
                        )
                        blocks = [
                            partial(
                                b,
                                chunk_size=tuned_chunk_size,
                                _attn_chunk_size=max(
                                    self.globals.chunk_size, tuned_chunk_size // 4
                                ),
                            )
                            for b in blocks
                        ]

                    if not self.training:
                        for i, b in enumerate(blocks):
                            b: TemplatePairStackBlock
                            b, attn_tri_s, attn_tri_e = b(t_pair)
                            attns[f"templ-tri_s-{i}"] = attn_tri_s
                            attns[f"templ-tri_e-{i}"] = attn_tri_e
                    else:
                        (t_pair,) = checkpoint_blocks(
                            blocks=blocks,
                            args=(t_pair,),
                            blocks_per_ckpt=None,
                        )

                    t = self.layer_norm(t_pair)
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # end of template pair stack
                    # =====================================================================

                    del t_pair

                    # add attention code
                    # [*, N, N, C_z]
                    t, attn = self.template_pointwise_att(
                        t,
                        z,
                        template_mask=batch["template_mask"].to(dtype=z.dtype),
                        use_lma=self.globals.use_lma,
                    )
                    attns["templ-pointwise"] = attn

                    t_mask = torch.sum(batch["template_mask"], dim=-1) > 0
                    # Append singletons
                    t_mask = t_mask.reshape(
                        *t_mask.shape, *([1] * (len(t.shape) - len(t_mask.shape)))
                    )

                    if inplace_safe:
                        t *= t_mask
                    else:
                        t = t * t_mask

                    template_embeds = {}

                    template_embeds.update({"template_pair_embedding": t})

                    del t

                    if self.config.template.embed_angles:
                        template_angle_feat = build_template_angle_feat(batch)

                        # [*, S_t, N, C_m]
                        a = self.template_angle_embedder(template_angle_feat)

                        template_embeds["template_angle_embedding"] = a
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # end of template stack
                    # =====================================================================

                    # [*, N, N, C_z]
                    z = add(
                        z,
                        template_embeds.pop("template_pair_embedding"),
                        inplace_safe,
                    )

                    if "template_angle_embedding" in template_embeds:
                        # [*, S = S_c + S_t, N, C_m]
                        m = torch.cat(
                            [m, template_embeds["template_angle_embedding"]], dim=-3
                        )

                        # [*, S, N]
                        torsion_angles_mask = feats["template_torsion_angles_mask"]
                        msa_mask = torch.cat(
                            [feats["msa_mask"], torsion_angles_mask[..., 2]], dim=-2
                        )

                # Embed extra MSA features + merge with pairwise embeddings
                if self.config.extra_msa.enabled:
                    # [*, S_e, N, C_e]
                    a = self.extra_msa_embedder(build_extra_msa_feat(feats))

                    if self.globals.offload_inference:
                        # To allow the extra MSA stack (and later the evoformer) to
                        # offload its inputs, we remove all references to them here
                        input_tensors = [a, z]
                        del a, z

                        # [*, N, N, C_z]
                        z = self.extra_msa_stack._forward_offload(
                            input_tensors,
                            msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                            chunk_size=self.globals.chunk_size,
                            use_lma=self.globals.use_lma,
                            pair_mask=pair_mask.to(dtype=m.dtype),
                            _mask_trans=self.config._mask_trans,
                        )

                        del input_tensors
                    else:
                        # =====================================================================
                        # start of extra MSA stack
                        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        # [*, N, N, C_z]
                        # z = self.extra_msa_stack(
                        #     a, z,
                        #     msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                        #     chunk_size=self.globals.chunk_size,
                        #     use_lma=self.globals.use_lma,
                        #     pair_mask=pair_mask.to(dtype=m.dtype),
                        #     inplace_safe=inplace_safe,
                        #     _mask_trans=self.config._mask_trans,
                        # )

                        checkpoint_fn = get_checkpoint_fn()
                        # blocks = self._prep_blocks(
                        #     m=a,
                        #     z=z,
                        #     chunk_size=self.globals.chunk_size,
                        #     use_lma=self.globals.use_lma,
                        #     msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                        #     pair_mask=pair_mask.to(dtype=m.dtype),
                        #     inplace_safe=inplace_safe,
                        #     _mask_trans=self.config._mask_trans,
                        # )
                        blocks = [
                            partial(
                                b,
                                msa_mask=feats["extra_msa_mask"].to(dtype=a.dtype),
                                pair_mask=pair_mask.to(dtype=a.dtype),
                                chunk_size=self.globals.chunk_size,
                                use_lma=self.globals.use_lma,
                                inplace_safe=inplace_safe,
                                _mask_trans=self.config._mask_trans,
                            )
                            for b in self.extra_msa_stack.blocks
                        ]

                        def clear_cache(b, *args, **kwargs):
                            torch.cuda.empty_cache()
                            return b(*args, **kwargs)

                        if self.extra_msa_stack.clear_cache_between_blocks:
                            blocks = [partial(clear_cache, b) for b in blocks]

                        if (
                            self.globals.chunk_size is not None
                            and self.extra_msa_stack.chunk_size_tuner is not None
                        ):
                            tuned_chunk_size = self.extra_msa_stack.chunk_size_tuner.tune_chunk_size(
                                representative_fn=blocks[0],
                                # Tensors cloned to avoid getting written to in-place
                                # A corollary is that chunk size tuning should be disabled for
                                # large N, when z gets really big
                                args=(
                                    a.clone(),
                                    z.clone(),
                                ),
                                min_chunk_size=self.globals.chunk_size,
                            )
                            blocks = [
                                partial(
                                    b,
                                    chunk_size=tuned_chunk_size,
                                    # A temporary measure to address torch's occasional
                                    # inability to allocate large tensors
                                    _attn_chunk_size=max(
                                        self.globals.chunk_size, tuned_chunk_size // 4
                                    ),
                                )
                                for b in blocks
                            ]

                        for i, b in enumerate(blocks):
                            b: ExtraMSABlock
                            if self.extra_msa_stack.ckpt and torch.is_grad_enabled():
                                a, z = checkpoint_fn(b, a, z)
                            else:
                                # add attention code
                                a, z, attn_row, attn_col, attn_tri_s, attn_tri_e = b(
                                    a, z
                                )
                                attns[f"extra_msa-row-{i}"] = attn_row
                                attns[f"extra_msa-col-{i}"] = attn_col
                                attns[f"extra_msa-tri_s-{i}"] = attn_tri_s
                                attns[f"extra_msa-tri_e-{i}"] = attn_tri_e
                        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        # end of extra MSA stack
                        # =====================================================================

                # =====================================================================
                # start of evoformer
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Run MSA + pair embeddings through the trunk of the network
                # m: [*, S, N, C_m]
                # z: [*, N, N, C_z]
                # s: [*, N, C_s]
                if self.globals.offload_inference:
                    input_tensors = [m, z]
                    del m, z
                    m, z, s = self.evoformer._forward_offload(
                        input_tensors,
                        msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                        pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                        chunk_size=self.globals.chunk_size,
                        use_lma=self.globals.use_lma,
                        _mask_trans=self.config._mask_trans,
                    )

                    del input_tensors
                else:
                    # m, z, s = self.evoformer(
                    #     m,
                    #     z,
                    #     msa_mask=msa_mask.to(dtype=m.dtype),
                    #     pair_mask=pair_mask.to(dtype=z.dtype),
                    #     chunk_size=self.globals.chunk_size,
                    #     use_lma=self.globals.use_lma,
                    #     use_flash=self.globals.use_flash,
                    #     inplace_safe=inplace_safe,
                    #     _mask_trans=self.config._mask_trans,
                    # )
                    # blocks = self._prep_blocks(
                    #     m=m,
                    #     z=z,
                    #     chunk_size=chunk_size,
                    #     use_lma=use_lma,
                    #     use_flash=use_flash,
                    #     msa_mask=msa_mask,
                    #     pair_mask=pair_mask,
                    #     inplace_safe=inplace_safe,
                    #     _mask_trans=_mask_trans,
                    # )
                    blocks = [
                        partial(
                            b,
                            msa_mask=msa_mask,
                            pair_mask=pair_mask,
                            chunk_size=self.globals.chunk_size,
                            use_lma=self.globals.use_lma,
                            use_flash=self.globals.use_flash,
                            inplace_safe=inplace_safe,
                            _mask_trans=self.config._mask_trans,
                        )
                        for b in self.evoformer.blocks
                    ]

                    if self.evoformer.clear_cache_between_blocks:

                        def block_with_cache_clear(block, *args, **kwargs):
                            torch.cuda.empty_cache()
                            return block(*args, **kwargs)

                        blocks = [partial(block_with_cache_clear, b) for b in blocks]

                    if (
                        self.globals.chunk_size is not None
                        and self.evoformer.chunk_size_tuner is not None
                    ):
                        assert not self.training
                        tuned_chunk_size = self.evoformer.chunk_size_tuner.tune_chunk_size(
                            representative_fn=blocks[0],
                            # We don't want to write in-place during chunk tuning runs
                            args=(
                                m.clone(),
                                z.clone(),
                            ),
                            min_chunk_size=self.globals.chunk_size,
                        )
                        blocks = [
                            partial(
                                b,
                                chunk_size=tuned_chunk_size,
                                # A temporary measure to address torch's occasional
                                # inability to allocate large tensors
                                _attn_chunk_size=max(
                                    self.globals.chunk_size, tuned_chunk_size // 4
                                ),
                            )
                            for b in blocks
                        ]

                    blocks_per_ckpt = self.evoformer.blocks_per_ckpt
                    if not torch.is_grad_enabled():
                        blocks_per_ckpt = None

                    if not torch.is_grad_enabled():
                        for b in blocks:
                            b: EvoformerBlock
                            # add attention code
                            m, z, attn_row, attn_col, attn_tri_s, attn_tri_e = b(m, z)
                            attns[f"evoformer-row-{i}"] = attn_row
                            attns[f"evoformer-col-{i}"] = attn_col
                            attns[f"evoformer-tri_s-{i}"] = attn_tri_s
                            attns[f"evoformer-tri_e-{i}"] = attn_tri_e
                    else:
                        m, z = checkpoint_blocks(
                            blocks,
                            args=(m, z),
                            blocks_per_ckpt=blocks_per_ckpt,
                        )

                    s = self.evoformer.linear(m[..., 0, :, :])

                    # return m, z, s
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # end of evoformer
                # =====================================================================

                outputs["msa"] = m[..., :n_seq, :, :]
                outputs["pair"] = z
                outputs["single"] = s

                del z

                # =====================================================================
                # start of structure module
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # Predict 3D structure
                # outputs["sm"] = self.structure_module(
                #     outputs,
                #     feats["aatype"],
                #     mask=feats["seq_mask"].to(dtype=s.dtype),
                #     inplace_safe=inplace_safe,
                #     _offload_inference=self.globals.offload_inference,
                # )
                sm = self.structure_module
                s = outputs["single"]

                mask = (feats["seq_mask"].to(dtype=s.dtype),)
                if mask is None:
                    # [*, N]
                    mask = s.new_ones(s.shape[:-1])

                # [*, N, C_s]
                s = sm.layer_norm_s(s)

                # [*, N, N, C_z]
                z = sm.layer_norm_z(outputs["pair"])

                z_reference_list = None
                if self.globals.offload_inference:
                    assert sys.getrefcount(outputs["pair"]) == 2
                    outputs["pair"] = outputs["pair"].cpu()
                    z_reference_list = [z]
                    z = None

                # [*, N, C_s]
                s_initial = s
                s = sm.linear_in(s)

                # [*, N]
                rigids = Rigid.identity(
                    s.shape[:-1],
                    s.dtype,
                    s.device,
                    self.training,
                    fmt="quat",
                )
                outputs_l = []
                for i in range(sm.no_blocks):
                    # ===============================================
                    # start of IPA
                    # +++++++++++++++++++++++++++++++++++++++++++++++
                    # [*, N, C_s]
                    # s = s + sm.ipa(
                    #     s,
                    #     z,
                    #     rigids,
                    #     mask,
                    #     inplace_safe=inplace_safe,
                    #     _offload_inference=self.globals.offload_inference,
                    #     _z_reference_list=z_reference_list
                    # )
                    if self.globals.offload_inference and inplace_safe:
                        z = z_reference_list
                    else:
                        z = [z]

                    #######################################
                    # Generate scalar and point activations
                    #######################################
                    # [*, N_res, H * C_hidden]
                    q = self.linear_q(s)
                    kv = self.linear_kv(s)

                    # [*, N_res, H, C_hidden]
                    q = q.view(q.shape[:-1] + (self.no_heads, -1))

                    # [*, N_res, H, 2 * C_hidden]
                    kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

                    # [*, N_res, H, C_hidden]
                    k, v = torch.split(kv, self.c_hidden, dim=-1)

                    # [*, N_res, H * P_q * 3]
                    q_pts = self.linear_q_points(s)

                    # This is kind of clunky, but it's how the original does it
                    # [*, N_res, H * P_q, 3]
                    q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
                    q_pts = torch.stack(q_pts, dim=-1)
                    q_pts = rigids[..., None].apply(q_pts)

                    # [*, N_res, H, P_q, 3]
                    q_pts = q_pts.view(
                        q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
                    )

                    # [*, N_res, H * (P_q + P_v) * 3]
                    kv_pts = self.linear_kv_points(s)

                    # [*, N_res, H * (P_q + P_v), 3]
                    kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
                    kv_pts = torch.stack(kv_pts, dim=-1)
                    kv_pts = rigids[..., None].apply(kv_pts)

                    # [*, N_res, H, (P_q + P_v), 3]
                    kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

                    # [*, N_res, H, P_q/P_v, 3]
                    k_pts, v_pts = torch.split(
                        kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
                    )

                    ##########################
                    # Compute attention scores
                    ##########################
                    # [*, N_res, N_res, H]
                    b = self.linear_b(z[0])

                    if self.globals.offload_inference:
                        assert sys.getrefcount(z[0]) == 2
                        z[0] = z[0].cpu()

                    # [*, H, N_res, N_res]
                    float16_enabled = torch.get_autocast_gpu_dtype() == torch.float16
                    if float16_enabled and torch.is_autocast_enabled():
                        with torch.cuda.amp.autocast(enabled=False):
                            a = torch.matmul(
                                permute_final_dims(
                                    q.float(), (1, 0, 2)
                                ),  # [*, H, N_res, C_hidden]
                                permute_final_dims(
                                    k.float(), (1, 2, 0)
                                ),  # [*, H, C_hidden, N_res]
                            )
                    else:
                        a = torch.matmul(
                            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
                            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
                        )
                    a *= math.sqrt(1.0 / (3 * self.c_hidden))
                    a += math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1))

                    # [*, N_res, N_res, H, P_q, 3]
                    pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
                    if inplace_safe:
                        pt_att *= pt_att
                    else:
                        pt_att = pt_att ** 2

                    # [*, N_res, N_res, H, P_q]
                    pt_att = sum(torch.unbind(pt_att, dim=-1))
                    head_weights = self.softplus(self.head_weights).view(
                        *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
                    )
                    head_weights = head_weights * math.sqrt(
                        1.0 / (3 * (self.no_qk_points * 9.0 / 2))
                    )
                    if inplace_safe:
                        pt_att *= head_weights
                    else:
                        pt_att = pt_att * head_weights

                    # [*, N_res, N_res, H]
                    pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
                    # [*, N_res, N_res]
                    square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
                    square_mask = self.inf * (square_mask - 1)

                    # [*, H, N_res, N_res]
                    pt_att = permute_final_dims(pt_att, (2, 0, 1))

                    # add attention code
                    if inplace_safe:
                        a += pt_att
                        del pt_att
                        a += square_mask.unsqueeze(-3)
                        # in-place softmax
                        attn_core_inplace_cuda.forward_(
                            a,
                            reduce(mul, a.shape[:-1]),
                            a.shape[-1],
                        )
                    else:
                        a = a + pt_att
                        a = a + square_mask.unsqueeze(-3)
                        a = self.softmax(a)
                    attns[f"ipa-attn-{i}"] = a.detach()

                    ################
                    # Compute output
                    ################
                    # [*, N_res, H, C_hidden]
                    o = torch.matmul(
                        a, v.transpose(-2, -3).to(dtype=a.dtype)
                    ).transpose(-2, -3)

                    # [*, N_res, H * C_hidden]
                    o = flatten_final_dims(o, 2)

                    # [*, H, 3, N_res, P_v]
                    if inplace_safe:
                        v_pts = permute_final_dims(v_pts, (1, 3, 0, 2))
                        o_pt = [
                            torch.matmul(a, v.to(a.dtype))
                            for v in torch.unbind(v_pts, dim=-3)
                        ]
                        o_pt = torch.stack(o_pt, dim=-3)
                    else:
                        o_pt = torch.sum(
                            (
                                a[..., None, :, :, None]
                                * permute_final_dims(v_pts, (1, 3, 0, 2))[
                                    ..., None, :, :
                                ]
                            ),
                            dim=-2,
                        )

                    # [*, N_res, H, P_v, 3]
                    o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
                    o_pt = rigids[..., None, None].invert_apply(o_pt)

                    # [*, N_res, H * P_v]
                    o_pt_norm = flatten_final_dims(
                        torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
                    )

                    # [*, N_res, H * P_v, 3]
                    o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

                    if self.globals.offload_inference:
                        z[0] = z[0].to(o_pt.device)

                    # [*, N_res, H, C_z]
                    o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))

                    # [*, N_res, H * C_z]
                    o_pair = flatten_final_dims(o_pair, 2)

                    # [*, N_res, C_s]
                    s = s + self.linear_out(
                        torch.cat(
                            (o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1
                        ).to(dtype=z[0].dtype)
                    )
                    # +++++++++++++++++++++++++++++++++++++++++++++++
                    # end of IPA
                    # ===============================================

                    s = sm.ipa_dropout(s)
                    s = sm.layer_norm_ipa(s)
                    s = sm.transition(s)

                    # [*, N]
                    rigids = rigids.compose_q_update_vec(sm.bb_update(s))

                    # To hew as closely as possible to AlphaFold, we convert our
                    # quaternion-based transformations to rotation-matrix ones
                    # here
                    backb_to_global = Rigid(
                        Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
                        rigids.get_trans(),
                    )

                    backb_to_global = backb_to_global.scale_translation(
                        sm.trans_scale_factor
                    )

                    # [*, N, 7, 2]
                    unnormalized_angles, angles = sm.angle_resnet(s, s_initial)

                    all_frames_to_global = sm.torsion_angles_to_frames(
                        backb_to_global,
                        angles,
                        feats["aatype"],
                    )

                    pred_xyz = sm.frames_and_literature_positions_to_atom14_pos(
                        all_frames_to_global,
                        feats["aatype"],
                    )

                    scaled_rigids = rigids.scale_translation(sm.trans_scale_factor)

                    preds = {
                        "frames": scaled_rigids.to_tensor_7(),
                        "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                        "unnormalized_angles": unnormalized_angles,
                        "angles": angles,
                        "positions": pred_xyz,
                        "states": s,
                    }

                    outputs_l.append(preds)

                    rigids = rigids.stop_rot_gradient()

                del z, z_reference_list

                if self.globals.offload_inference:
                    outputs["pair"] = outputs["pair"].to(s.device)

                outputs_l = dict_multimap(torch.stack, outputs_l)
                outputs_l["single"] = s
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # end of structure module
                # =====================================================================

                outputs["sm"] = outputs_l
                outputs["final_atom_positions"] = atom14_to_atom37(
                    outputs["sm"]["positions"][-1], feats
                )
                outputs["final_atom_mask"] = feats["atom37_atom_exists"]
                outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

                # Save embeddings for use during the next recycling iteration

                # [*, N, C_m]
                m_1_prev = m[..., 0, :, :]

                # [*, N, N, C_z]
                z_prev = outputs["pair"]

                # [*, N, 3]
                x_prev = outputs["final_atom_positions"]

                # return outputs, m_1_prev, z_prev, x_prev

                if not is_final_iter:
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev

        # Run auxiliary heads
        outputs.update(self.aux_heads(outputs))
        outputs["attns"] = attns

        return outputs

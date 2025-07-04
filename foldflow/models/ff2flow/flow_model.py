from typing import Dict, Tuple, Optional

import torch
import logging

from torch_geometric.data import Data, Batch

from foldflow.data import all_atom
from foldflow.models.ff2flow.adapters import (
    MACEEncoderToTrunkNetwork,
    ProjectConcatRepresentation,
    SequenceToTrunkNetwork,
    TrunkToDecoderNetwork,
)
from foldflow.models.ff2flow.ff2_dependencies import FF2Dependencies
from foldflow.models.ff2flow.structure_network import FF2StructureNetwork
from foldflow.models.ff2flow.trunk import FF2TrunkTransformer
from foldflow.models.components.sequence.frozen_esm import FrozenEsmModel
from foldflow.models.components.structure.mace import MACEModel
from openfold.utils import rigid_utils as ru
from torch import nn
from foldflow.models.components.sequence.frozen_esm import ESM_REGISTRY
from foldflow.models.se3_fm import SE3FlowMatcher
from foldflow.utils.graph_helpers import find_isolated_nodes, build_graph


class FF2Model(nn.Module):
    def __init__(
        self,
        config,
        flow_matcher: SE3FlowMatcher,
        bb_encoder: FF2StructureNetwork,
        bb_mace_encoder: Optional[MACEModel],
        bb_decoder: FF2StructureNetwork,
        seq_encoder: FrozenEsmModel,
        sequence_to_trunk_network: SequenceToTrunkNetwork,
        bb_mace_encoder_to_trunk_network: MACEEncoderToTrunkNetwork,
        combiner_network: ProjectConcatRepresentation,
        trunk_network: Optional[FF2TrunkTransformer],
        trunk_to_decoder_network: TrunkToDecoderNetwork,
        time_embedder,
    ):
        super().__init__()
        self.config = config
        self.flow_matcher = flow_matcher
        self.bb_encoder = bb_encoder
        self.bb_mace_encoder = bb_mace_encoder
        self.bb_decoder = bb_decoder
        self.seq_encoder = seq_encoder
        self.sequence_to_trunk_network = sequence_to_trunk_network
        self.bb_mace_encoder_to_trunk_network = bb_mace_encoder_to_trunk_network
        self.combiner_network = combiner_network
        self.trunk_network = trunk_network
        self.trunk_to_decoder_network = trunk_to_decoder_network
        self.time_embedder = time_embedder

        self._is_conditional_generation = False
        self._is_scaffolding_generation = False

        self._debug = config.experiment.debug
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

    @property
    def is_cond_seq(self):
        return self.config.is_cond_seq

    @classmethod
    def from_dependencies(cls, dependencies: FF2Dependencies):
        return cls(
            config=dependencies.config,
            flow_matcher=dependencies.flow_matcher,
            bb_encoder=dependencies.bb_encoder,
            bb_mace_encoder=dependencies.bb_mace_encoder,
            bb_decoder=dependencies.bb_decoder,
            seq_encoder=dependencies.seq_encoder,
            sequence_to_trunk_network=dependencies.sequence_to_trunk_network,
            bb_mace_encoder_to_trunk_network=dependencies.bb_mace_encoder_to_trunk_network,
            combiner_network=dependencies.combiner_network,
            trunk_network=dependencies.trunk_network,
            trunk_to_decoder_network=dependencies.trunk_to_decoder_network,
            time_embedder=dependencies.time_embedder,
        )

    @classmethod
    def from_ckpt(cls, ckpt: Dict[str, torch.Tensor], deps: FF2Dependencies):
        _prefix_to_remove = "vectorfield_network."
        ckpt["state_dict"] = ckpt["model"]
        ckpt["state_dict"] = {k.replace(_prefix_to_remove, ""): v for k, v in ckpt["state_dict"].items()}
        model = cls.from_dependencies(deps)
        if model.bb_mace_encoder is not None:
            model._logger.info("MACE encoder is ON in the model.")
        else:
            model._logger.info("MACE encoder is OFF in the model.")
        # TODO: fix the improper saving of the ESM model to make the assertion work.
        if "esm_model" not in ckpt:
            model._logger.warning(
                "The checkpoint does not contain 'esm_model' key. "
                "There's no way to verify the exact ESM model version used for training. Use with caution."
            )
        else:
            ckpt_lm_name = ckpt["esm_model"]
            assert (
                deps.config.model.esm2_model_key == ckpt_lm_name
            ), f"Model trained with different ESM2 {ckpt_lm_name}, but got {deps.config.model.esm2_model_key=}"

        # Fix multi-GPU ckpt loading.
        is_parallel_ckpt = list(ckpt["state_dict"].keys())[0].startswith("module.")
        is_parallel_model = isinstance(model, torch.nn.DataParallel)

        if is_parallel_ckpt and not is_parallel_model:
            ckpt["state_dict"] = {k.replace("module.", "", 1): v for k, v in ckpt["state_dict"].items()}

        cls._add_esm_to_ckpt(model, ckpt)
        model.load_state_dict(ckpt["state_dict"])
        return model

    @staticmethod
    def _add_esm_to_ckpt(model, ckpt: Dict[str, torch.Tensor]) -> None:
        # Determine if we are in a multi-GPU checkpoint
        prefix = "module." if list(ckpt["state_dict"].keys())[0].startswith("module.") else ""

        for k, v in model.seq_encoder.state_dict().items():
            if k.startswith("esm."):
                ckpt["state_dict"][f"{prefix}seq_encoder.{k}"] = v

    def _get_vectorfields(
        self,
        pred_rigids: torch.Tensor,
        init_rigids: torch.Tensor,
        t: torch.Tensor,
        res_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, rot_vectorfield = self.flow_matcher.calc_rot_vectorfield(
            pred_rigids.get_rots().get_rot_mats(),
            init_rigids.get_rots().get_rot_mats(),
            t,
        )
        rot_vectorfield = rot_vectorfield * res_mask[..., None, None]
        trans_vectorfield = self.flow_matcher.calc_trans_vectorfield(
            pred_rigids.get_trans(),
            init_rigids.get_trans(),
            t[:, None, None],
            scale=True,
        )
        trans_vectorfield = trans_vectorfield * res_mask[..., None]
        return rot_vectorfield, trans_vectorfield

    def _make_seq_mask_pattern(self, batch):
        aatype = batch["aatype"]
        if self.is_scaffolding_generation:
            return 1.0 - batch["fixed_mask_seq"]
        if self._is_conditional_generation:
            # no masking during conditional generation
            return torch.zeros_like(aatype)

        if not self.training:
            # mask the entire sequence for inference
            return torch.ones_like(aatype)

        pattern = torch.zeros_like(aatype)
        rows_to_mask = (torch.rand(aatype.shape[0]) < self.config.model.p_mask_sequence).to(aatype.device)
        pattern[rows_to_mask] = 1
        return pattern

    @property
    def is_conditional_generation(self) -> bool:
        return self._is_conditional_generation

    def conditional_generation(self):
        self.eval()
        self._is_conditional_generation = True

    @property
    def is_scaffolding_generation(self) -> bool:
        return self._is_scaffolding_generation

    def scaffolding_generation(self):
        self._is_scaffolding_generation = True

    def train(self, is_training):
        # nn.Module.eval() calls self.train(False)
        self._is_conditional_generation = False
        super().train(is_training)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # TODO: verify the return type.
        device = batch["rigids_t"].device
        init_rigids = ru.Rigid.from_tensor_7(batch["rigids_t"])
        t = batch["t"]

        bb_mask = batch["res_mask"].type(torch.float32)  # [B, N]
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        # Sequence representations.
        seq_mask_pattern = self._make_seq_mask_pattern(batch)

        seq_emb_s, seq_emb_z = self.seq_encoder(
            batch["aatype"],
            batch["chain_idx"],
            attn_mask=batch["res_mask"],
            seq_mask=seq_mask_pattern,
        )
        seq_emb_s, seq_emb_z = seq_emb_s.to(device), seq_emb_z.to(device)
        # Processing of the sequence emb (trainable). # LN and Lin. layers.
        seq_emb_s, seq_emb_z = self.sequence_to_trunk_network(seq_emb_s, seq_emb_z, batch["seq_idx"], batch["res_mask"])

        # Structure encoder representations.
        bb_encoder_output = self.bb_encoder(
            res_mask=batch["res_mask"],
            fixed_mask=batch["fixed_mask"],
            seq_idx=batch["seq_idx"],
            chain_idx=batch["chain_idx"],
            t=t,
            rigids_t=batch["rigids_t"],
            unscale_rigids=False,
            self_conditioning_ca=batch["sc_ca_t"],
        )
        bb_emb_s = bb_encoder_output["single_emb"]
        bb_emb_z = bb_encoder_output["pair_emb"]
        rigids_updated = bb_encoder_output["rigids"]
        init_single_embed = bb_encoder_output["init_single_embed"]
        init_pair_embed = bb_encoder_output["init_pair_embed"]

        # If MACE encoder is used, produce MACE representations.
        has_self_conditioning_output = not torch.all(batch["sc_ca_t"] == 0.0)

        if self.bb_mace_encoder is not None and has_self_conditioning_output:
            # Create a graph using only CA positions.
            ca_pos = batch["sc_ca_t"].to(torch.float32)
            res_idx = batch["residue_index"] if "residue_index" in batch else batch["seq_idx"]
            atoms = torch.zeros_like(ca_pos[..., 0]).to(torch.long)  # Dummy atomic type, as MACE requires it.
            data = Batch.from_data_list(
                [Data(atoms=atoms[i], pos=ca_pos[i], res_idx=res_idx[i]) for i in range(ca_pos.shape[0])]
            )
            # Create edges based on the spatial distances and KNN between CA atoms, but not connecting
            # CA atoms of the neighbouring residues.
            max_num_edges = self.config.model.graph.max_squared_res_ratio * (data.num_nodes**2)
            data.edge_index = build_graph(data, max_edges=max_num_edges, min_residue_distance=5, radius=5, k=10)

            # Check for isolated nodes.
            assert (
                len(find_isolated_nodes(data.num_nodes, data.edge_index)) == 0
            ), "Some nodes are isolated in the graph."

            # Compute MACE representations. The result contains only updated O(3)-equivariant node features,
            # the pairwise features are encoded in the node features --> we have only single representation here.
            bb_mace_emb_s = self.bb_mace_encoder(data)
            if self._debug:
                self._logger.info(f"The number of edges in the graph: {data.edge_index.shape[1]}")

            # Process MACE representations (Here we lose equivariance of the features)
            bb_mace_emb_s = self.bb_mace_encoder_to_trunk_network(bb_mace_emb_s)

        else:
            bb_mace_emb_s_dim = self.config.model.bb_mace_encoder_to_block.single_dim
            bb_mace_emb_s = torch.zeros(
                batch["sc_ca_t"].shape[:-1] + torch.Size([bb_mace_emb_s_dim]),
                device=bb_emb_s.device,
                dtype=bb_emb_s.dtype,
            )

        # Log norms for debugging.
        if self._debug:
            self._logger.info(f"Norm of seq_emb_s: {seq_emb_s.norm()}")
            self._logger.info(f"Norm of seq_emb_z: {seq_emb_z.norm()}")
            self._logger.info(f"Norm of bb_emb_s: {bb_emb_s.norm()}")
            self._logger.info(f"Norm of bb_emb_z: {bb_emb_z.norm()}")
            if has_self_conditioning_output and self.bb_mace_encoder:
                self._logger.info(f"Norm of bb_mace_emb_s: {bb_mace_emb_s.norm()}")

        # Representations combiner
        single_representation = {"bb": bb_emb_s, "seq": seq_emb_s, "bb_mace": bb_mace_emb_s}
        pair_representation = {"bb": bb_emb_z, "seq": seq_emb_z}
        single_embed, pair_embed = self.combiner_network(single_representation, pair_representation)

        # Evoformer or identity.
        if self.trunk_network:
            single_embed, pair_embed = self.trunk_network(single_embed, pair_embed, mask=batch["res_mask"].float())

        # Update representations dim for decoder. Uses just one Linear and LN layer.
        single_embed, pair_embed = self.trunk_to_decoder_network(single_embed, pair_embed)

        # Add a skip connection and average the result.
        single_embed = 0.5 * (single_embed + init_single_embed)
        pair_embed = 0.5 * (pair_embed + init_pair_embed)

        # edge and node masking
        single_embed = single_embed * bb_mask[..., None]
        pair_embed = pair_embed * edge_mask[..., None]

        # update the rigids with the new single and pair representation.
        bb_decoder_output = self.bb_decoder(
            res_mask=batch["res_mask"],
            fixed_mask=batch["fixed_mask"],
            t=t,
            single_embed=single_embed,
            pair_embed=pair_embed,
            rigids_t=rigids_updated.to_tensor_7(),
            unscale_rigids=False,
        )
        rigids_updated = bb_decoder_output["rigids"]
        psi = bb_decoder_output["psi"]
        if self._is_scaffolding_generation:
            mask = batch["fixed_mask"][:, :, None]
            gt_psi = batch["torsion_angles_sin_cos"][..., 2, :]
            psi = psi * (1 - mask) + gt_psi * mask

        res_mask = batch["res_mask"].type(torch.float32)
        rot_vectorfield, trans_vectorfield = self._get_vectorfields(rigids_updated, init_rigids, t, res_mask)
        model_out: Dict[str, torch.Tensor] = {}
        model_out["rot_vectorfield"] = rot_vectorfield
        model_out["trans_vectorfield"] = trans_vectorfield
        model_out["psi"] = psi
        model_out["rigids"] = rigids_updated.to_tensor_7()
        bb_representations = all_atom.compute_backbone(rigids_updated, psi)
        model_out["atom37"] = bb_representations[0].to(device)
        model_out["atom14"] = bb_representations[-1].to(device)

        return model_out

import torch

from e3nn import o3, nn
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict

from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from typing import Optional, Dict

from foldflow.models.components.structure.modules.irreps_tools import reshape_irreps
from foldflow.models.components.structure.modules.blocks import (
    EquivariantProductBasisBlock,
    RadialEmbeddingBlock,
)
from foldflow.models.components.structure.layers.tfn_layer import TensorProductConvLayer


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MACEConfig:
    """
    Parameters:
    - r_max (float): Maximum distance for Bessel basis functions (default: 10.0)
    - num_bessel (int): Number of Bessel basis functions (default: 8)
    - num_polynomial_cutoff (int): Number of polynomial cutoff basis functions (default: 5)
    - max_ell (int): Maximum degree of spherical harmonics basis functions (default: 2)
    - correlation (int): Local correlation order = body order - 1 (default: 3)
    - num_layers (int): Number of layers in the model (default: 5)
    - emb_dim (int): Scalar feature embedding dimension (default: 64)
    - hidden_irreps (Optional[e3nn.o3.Irreps]): Hidden irreps (default: None)
    - mlp_dim (int): Dimension of MLP for computing tensor product weights (default: 256)
    - in_dim (int): Input dimension of the model (default: 1)
    - out_dim (int): Output dimension of the model (default: 1)
    - aggr (str): Aggregation method to be used (default: "sum")
    - pool (str): Global pooling method to be used (default: "sum")
    - batch_norm (bool): Whether to use batch normalization (default: True)
    - residual (bool): Whether to use residual connections (default: True)
    - equivariant_pred (bool): Whether it is an equivariant prediction task (default: False)
    - as_encoder (bool): Whether to use the model as an encoder (default: True)
    - encoder_dim (int): Dimension of the encoder output (default: 256)

    Note:
    - If `hidden_irreps` is None, the irreps for the intermediate features are computed
      using `emb_dim` and `max_ell`.
    - The `equivariant_pred` parameter determines whether it is an equivariant prediction task.
      If set to True, equivariant prediction will be performed.
    """

    r_max: float = 10.0
    num_bessel: int = 8
    num_polynomial_cutoff: int = 5
    max_ell: int = 2
    correlation: int = 3
    num_layers: int = 5
    emb_dim: int = 64
    hidden_irreps: Optional[o3.Irreps] = None
    mlp_dim: int = 256
    in_dim: int = 1
    out_dim: int = 1
    aggr: str = "sum"
    pool: str = "sum"
    batch_norm: bool = True
    residual: bool = True
    equivariant_pred: bool = True
    as_encoder: bool = True
    encoder_dim: int = 256


class MACEModel(torch.nn.Module):
    """
    MACE model from "MACE: Higher Order Equivariant Message Passing Neural Networks".
    """

    def __init__(
        self,
        conf: MACEConfig,
    ):
        super().__init__()
        self._conf = conf
        self.emb_dim = conf.emb_dim
        self.equivariant_pred = conf.equivariant_pred
        self.as_encoder = conf.as_encoder

        # Edge embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=conf.r_max,
            num_bessel=conf.num_bessel,
            num_polynomial_cutoff=conf.num_polynomial_cutoff,
        )
        sh_irreps = o3.Irreps.spherical_harmonics(conf.max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization="component")

        # Embedding lookup for initial node features
        self.emb_in = torch.nn.Embedding(conf.in_dim, conf.emb_dim)

        # Set hidden irreps if none are provided
        if conf.hidden_irreps is None:
            hidden_irreps = (sh_irreps * conf.emb_dim).sort()[0].simplify()
            # Note: This defaults to O(3) equivariant layers
            # It is possible to use SO(3) equivariance by passing the appropriate irreps

        self.convs = torch.nn.ModuleList()
        self.prods = torch.nn.ModuleList()
        self.reshapes = torch.nn.ModuleList()

        # First layer: scalar only -> tensor
        self.convs.append(
            TensorProductConvLayer(
                in_irreps=o3.Irreps(f"{conf.emb_dim}x0e"),
                out_irreps=hidden_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                mlp_dim=conf.mlp_dim,
                aggr=conf.aggr,
                batch_norm=conf.batch_norm,
                gate=False,
            )
        )
        self.reshapes.append(reshape_irreps(hidden_irreps))
        self.prods.append(
            EquivariantProductBasisBlock(
                node_feats_irreps=hidden_irreps,
                target_irreps=hidden_irreps,
                correlation=conf.correlation,
                element_dependent=False,
                num_elements=conf.in_dim,
                use_sc=conf.residual,
            )
        )

        # Intermediate layers: tensor -> tensor
        for _ in range(conf.num_layers - 1):
            self.convs.append(
                TensorProductConvLayer(
                    in_irreps=hidden_irreps,
                    out_irreps=hidden_irreps,
                    sh_irreps=sh_irreps,
                    edge_feats_dim=self.radial_embedding.out_dim,
                    mlp_dim=conf.mlp_dim,
                    aggr=conf.aggr,
                    batch_norm=conf.batch_norm,
                    gate=False,
                )
            )
            self.reshapes.append(reshape_irreps(hidden_irreps))
            self.prods.append(
                EquivariantProductBasisBlock(
                    node_feats_irreps=hidden_irreps,
                    target_irreps=hidden_irreps,
                    correlation=conf.correlation,
                    element_dependent=False,
                    num_elements=conf.in_dim,
                    use_sc=conf.residual,
                )
            )

        if conf.as_encoder:
            # If used as an encoder, reduce the output dimension and create an equivariant linear encoder layer
            # that would project the hidden irreps to the final scalar irreps.
            # We use only scalars because further layers are not equivariant.
            final_irreps = o3.Irreps(f"{conf.encoder_dim}x0e")

            self.encoder_head = torch.nn.ModuleList()
            self.encoder_head.append(
                o3.Linear(
                    irreps_in=hidden_irreps,
                    irreps_out=final_irreps,
                )
            )
            if conf.batch_norm:
                self.encoder_head.append(nn.BatchNorm(final_irreps))

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[conf.pool]

        if self.equivariant_pred:
            # Linear predictor for equivariant tasks using geometric features
            self.pred = torch.nn.Linear(hidden_irreps.dim, conf.out_dim)
        else:
            # MLP predictor for invariant tasks using only scalar features
            self.pred = torch.nn.Sequential(
                torch.nn.Linear(conf.emb_dim, conf.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(conf.emb_dim, conf.out_dim),
            )

    def forward(self, batch):
        # Node embedding
        h = self.emb_in(batch.atoms)  # (n,) -> (n, d)

        # Edge features
        vectors = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]

        edge_sh = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        for conv, reshape, prod in zip(self.convs, self.reshapes, self.prods):
            # Message passing layer
            h_update = conv(node_attr=h, edge_index=batch.edge_index, edge_sh=edge_sh, edge_feat=edge_feats)

            # Update node features
            sc = F.pad(h, (0, h_update.shape[-1] - h.shape[-1]))  # skip connection
            h = prod(node_feats=reshape(h_update), sc=sc, node_attrs=None)

        if self.as_encoder:
            # If used as an encoder, apply the linear layer and batch normalization (if applicable)
            for layer in self.encoder_head:
                h = layer(h)
            return h.view(batch.num_graphs, -1, h.shape[-1])  # (n, final_irreps.dim)

        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)

        if not self.equivariant_pred:
            # Select only scalars for invariant prediction
            out = out[:, : self.emb_dim]

        return self.pred(out)  # (batch_size, out_dim)

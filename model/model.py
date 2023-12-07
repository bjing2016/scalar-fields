from e3nn import o3
import torch
from torch import nn
from torch_cluster import radius_graph
import numpy as np
from .utils import (
    GaussianSmearing,
    Encoder,
    TensorProductConvLayer,
)


class TensorProductConvModel(torch.nn.Module):
    def __init__(self, args, node_features, edge_features, atom_features=None):
        super(TensorProductConvModel, self).__init__()

        self.args = args

        self.sh_irreps = o3.Irreps([(1, (l, 1)) for l in range(args.order + 1)])
        self.feature_irreps = o3.Irreps(
            [(args.ns if l == 0 else args.nv, (l, 1)) for l in range(args.order + 1)]
        )

        self.edge_features = edge_features

        self.out_irreps = o3.Irreps(
            [(args.num_rbf * args.num_channels, (l, 1)) for l in range(args.order + 1)]
        )
        self.register_buffer(
            "sorter", self.get_sorter(args.order, args.num_rbf * args.num_channels)
        )

        self.node_encoder = Encoder(args.ns, node_features)
        self.node_embedding = nn.Sequential(
            nn.Linear(args.ns, args.ns),  # fix
            nn.ReLU(),
            nn.Linear(args.ns, args.ns),
        )

        if atom_features:
            self.a_node_encoder = Encoder(args.ns, atom_features)
            self.a_node_embedding = nn.Sequential(
                nn.Linear(args.ns, args.ns),  # fix
                nn.ReLU(),
                nn.Linear(args.ns, args.ns),
            )

        if edge_features > 0:
            self.edge_embedding = nn.Sequential(
                nn.Linear(edge_features, args.ns),
                nn.ReLU(),
                nn.Linear(args.ns, args.ns),
            )

        self.radius_embedding = nn.Sequential(
            nn.Linear(args.radius_emb_dim, args.ns),
            nn.ReLU(),
            nn.Linear(args.ns, args.ns),
        )
        self.distance_expansion = GaussianSmearing(0.0, args.radius_emb_max, args.radius_emb_dim)
        conv_layers = []
        if atom_features:
            a_conv_layers = []
            cross_conv_layers = []
        for i in range(args.conv_layers):
            layer = lambda residual: TensorProductConvLayer(
                in_irreps=self.feature_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=self.feature_irreps,
                n_edge_features=3 * args.ns,
                batch_norm=False,
                residual=residual,
                hidden_features=args.fc_dim,
            )
            conv_layers.append(layer(True))
            if atom_features:
                a_conv_layers.append(layer(True))
                cross_conv_layers.append(layer(False))

        self.conv_layers = nn.ModuleList(conv_layers)
        if atom_features:
            self.a_conv_layers = nn.ModuleList(a_conv_layers)
            self.cross_conv_layers = nn.ModuleList(cross_conv_layers)
        self.linear_final = o3.Linear(self.feature_irreps, self.out_irreps)

    def get_sorter(self, order, num_channels):
        ### getting the reindexer
        idx = []
        ord = []
        m = []
        for l in range(order + 1):
            for n in range(num_channels):
                idx.extend([n] * (2 * l + 1))
                ord.extend(list(range(l**2, (l + 1) ** 2)))
        idx = np.array(idx)
        ord = np.array(ord)
        sorter = np.argsort(idx * (ord.max() + 1) + ord)
        return torch.from_numpy(sorter)

    def all_atom_forward(self, data):
        r_pos = data["receptor"].pos
        a_pos = data["atom"].pos

        r_node_attr = self.node_embedding(self.node_encoder(data["receptor"].node_attr))
        a_node_attr = self.a_node_embedding(self.a_node_encoder(data["atom"].node_attr))

        r_edge_index = data["receptor", "receptor"].edge_index
        a_edge_index = data["atom", "atom"].edge_index
        try:
            r_batch = data["receptor"].batch
            a_batch = data["atom"].batch
        except:
            r_batch = None
            a_batch = None

        r_edge_index, r_edge_attr, r_edge_sh = self.build_conv_graph(
            r_pos, None, r_edge_index, r_batch, radius=False
        )
        a_edge_index, a_edge_attr, a_edge_sh = self.build_conv_graph(
            a_pos, None, a_edge_index, a_batch, radius=False
        )

        ar_edge_index, ar_edge_attr, ar_edge_sh = self.build_cross_conv_graph(data)

        r_node_attr = nn.functional.pad(r_node_attr, [0, self.feature_irreps.dim - self.args.ns])
        a_node_attr = nn.functional.pad(a_node_attr, [0, self.feature_irreps.dim - self.args.ns])

        for layer, a_layer, cross_layer in zip(
            self.conv_layers, self.a_conv_layers, self.cross_conv_layers
        ):
            ns = self.args.ns

            a_src, a_dst = a_edge_index
            a_edge_attr_ = torch.cat(
                [a_edge_attr, a_node_attr[a_src, :ns], a_node_attr[a_dst, :ns]], -1
            )
            a_node_attr = layer(a_node_attr, a_edge_index, a_edge_attr_, a_edge_sh)

            r_src, r_dst = r_edge_index
            r_edge_attr_ = torch.cat(
                [r_edge_attr, r_node_attr[r_src, :ns], r_node_attr[r_dst, :ns]], -1
            )
            r_node_attr = a_layer(r_node_attr, r_edge_index, r_edge_attr_, r_edge_sh)

            ar_src, ar_dst = ar_edge_index
            ar_edge_attr_ = torch.cat(
                [ar_edge_attr, a_node_attr[ar_src, :ns], r_node_attr[ar_dst, :ns]], -1
            )
            r_node_attr = r_node_attr + cross_layer(
                a_node_attr,
                ar_edge_index.flip(0),
                ar_edge_attr_,
                ar_edge_sh,
                out_nodes=r_node_attr.shape[0],
            )

        resi_out = self.linear_final(r_node_attr)
        resi_out = resi_out[:, self.sorter]
        resi_out = resi_out.view(
            -1, self.args.num_channels, self.args.num_rbf, (self.args.order + 1) ** 2
        )
        return resi_out

    def forward(self, data, key="ligand", radius=True, all_atoms=False):
        if all_atoms:
            return self.all_atom_forward(data)

        pos = data[key].pos

        node_attr = self.node_embedding(self.node_encoder(data[key].node_attr))
        if self.edge_features > 0:
            edge_attr = self.edge_embedding(data[key, key].edge_attr)
        else:
            edge_attr = None
        edge_index = data[key, key].edge_index
        try:
            batch = data[key].batch
        except:
            batch = None

        edge_index, edge_attr, edge_sh = self.build_conv_graph(
            pos, edge_attr, edge_index, batch, radius=radius
        )

        node_attr = nn.functional.pad(node_attr, [0, self.feature_irreps.dim - self.args.ns])
        src, dst = edge_index

        for layer in self.conv_layers:
            ns = self.args.ns
            edge_attr_ = torch.cat([edge_attr, node_attr[src, :ns], node_attr[dst, :ns]], -1)
            node_attr = layer(node_attr, edge_index, edge_attr_, edge_sh)

        resi_out = self.linear_final(node_attr)
        resi_out = resi_out[:, self.sorter]
        resi_out = resi_out.view(
            -1, self.args.num_channels, self.args.num_rbf, (self.args.order + 1) ** 2
        )
        return resi_out
        # return resi_out.permute(2, 0, 1, 3)  # channels first

    def build_conv_graph(self, pos, edge_attr, edge_index, batch=None, radius=True):
        if radius:
            radius_edges = radius_graph(
                pos, self.args.radius_emb_max, batch, max_num_neighbors=1000
            )
            edge_index = torch.cat([edge_index, radius_edges], 1).long()
            if edge_attr is not None:
                edge_attr = torch.nn.functional.pad(edge_attr, [0, 0, 0, radius_edges.shape[-1]])

        src, dst = edge_index
        edge_vec = pos[src.long()] - pos[dst.long()]
        edge_length_emb = self.distance_expansion(edge_vec.norm(dim=-1))

        if edge_attr is not None:
            edge_attr = edge_attr + self.radius_embedding(edge_length_emb)
        else:
            edge_attr = self.radius_embedding(edge_length_emb)

        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        ).float()
        return edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data):
        # ATOM to RECEPTOR
        edge_index = data["atom", "receptor"].edge_index
        src, dst = edge_index
        edge_vec = data["receptor"].pos[dst.long()] - data["atom"].pos[src.long()]
        edge_length_emb = self.distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = self.radius_embedding(edge_length_emb)
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )
        return edge_index, edge_attr, edge_sh

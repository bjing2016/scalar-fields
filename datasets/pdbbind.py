#####
# Modified from https://github.com/gcorso/DiffDock/blob/main/datasets/pdbbind.py
#####

from torch_geometric.data import Dataset, HeteroData
from utils.logging import get_logger

logger = get_logger(__name__)

AVOGADRO = 6.02214076e23

from .process_mols import (
    read_mol,
    get_calpha_graph,
    get_lig_graph,
    extract_receptor_structure,
    parse_receptor,
)


class PDBBind(Dataset):
    def __init__(self, args, split_path, torsion_noise=False,
                 return_all=False, esmfold=False):
        super(PDBBind, self).__init__()

        self.args = args
        self.return_all = return_all
        self.esmfold = esmfold
        self.split = list(map(lambda x: x.strip(), open(split_path)))

    def len(self):
        return len(self.split)

    def get(self, idx):
        if self.args.overfit:
            return self.get_complex(0)
        cmplx = self.get_complex(idx)
        if self.return_all:
            return cmplx
        if not cmplx:
            return self.get((idx + 1) % len(self))
        return cmplx
        
    def get_complex(self, idx):

        complex_graph = HeteroData()
        complex_graph.name = name = self.split[idx]
        complex_graph.meta = {'name': name, 'idx': idx}

        rec_model = parse_receptor(name, self.args.pdbbind_dir, esmfold=self.esmfold)
        complex_graph.rdkit_mol = lig = read_mol(self.args.pdbbind_dir, name, remove_hs=True)
        complex_graph.full_rdkit_mol = read_mol(self.args.pdbbind_dir, name, remove_hs=False)
        if lig is None:
            logger.warning(f"Skipping {name}, RDKit failure")
            return
            
        if lig.GetNumHeavyAtoms() > self.args.max_lig_size:
            logger.warning(f"Skipping {name} lig_heavy_atoms={lig.GetNumHeavyAtoms()}")
            return

        get_lig_graph(lig, complex_graph, get_masks=False)
        complex_graph.rdkit_mol = lig
        
        smi = complex_graph['meta']['smiles']
        
        receptor = extract_receptor_structure(rec_model, lm_embedding_chains=None)

        protlen = receptor['meta']['valid_residues']
        if protlen > self.args.max_protein_len:
            logger.warning(f"Skipping {name}, protein_len={protlen}")
            return
        complex_graph.meta |= receptor['meta']

        get_calpha_graph(
            receptor,
            complex_graph,
            cutoff=self.args.receptor_radius,
            max_neighbor=self.args.c_alpha_max_neighbors,
            all_atoms=self.args.all_atoms,
            atom_radius=self.args.atom_radius,
            atom_max_neighbors=self.args.atom_max_neighbors,
        )

        if self.args.ablate_chemistry:
            complex_graph['ligand'].node_attr *= 0
            complex_graph['ligand', 'ligand'].edge_attr *= 0
            complex_graph['receptor'].node_attr *= 0
            if self.args.all_atoms:
                complex_graph['atom'].node_attr *= 0
                

        return complex_graph



#####
# Modified from https://github.com/gcorso/DiffDock/blob/main/datasets/process_mols.py
#####

import copy, os, warnings, torch
import numpy as np
import scipy.spatial as spa
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import Polypeptide
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
from rdkit.Geometry import Point3D
from torch_cluster import radius_graph
import torch.nn.functional as F
from spyrmsd import rmsd, molecule
from utils.logging import get_logger
logger = get_logger(__name__)

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


biopython_parser = PDBParser()
periodic_table = GetPeriodicTable()
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OCTAHEDRAL",
        "CHI_SQUAREPLANAR",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_numring_list": [0, 1, 2, 3, 4, 5, 6, "misc"],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring3_list": [False, True],
    "possible_is_in_ring4_list": [False, True],
    "possible_is_in_ring5_list": [False, True],
    "possible_is_in_ring6_list": [False, True],
    "possible_is_in_ring7_list": [False, True],
    "possible_is_in_ring8_list": [False, True],
    "possible_amino_acids": [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "HIP",
        "HIE",
        "TPO",
        "HID",
        "LEV",
        "MEU",
        "PTR",
        "GLV",
        "CYT",
        "SEP",
        "HIZ",
        "CYM",
        "GLM",
        "ASQ",
        "TYS",
        "CYX",
        "GLZ",
        "misc",
    ],
    "possible_atom_type_2": [
        "C*",
        "CA",
        "CB",
        "CD",
        "CE",
        "CG",
        "CH",
        "CZ",
        "N*",
        "ND",
        "NE",
        "NH",
        "NZ",
        "O*",
        "OD",
        "OE",
        "OG",
        "OH",
        "OX",
        "S*",
        "SD",
        "SG",
        "misc",
    ],
    "possible_atom_type_3": [
        "C",
        "CA",
        "CB",
        "CD",
        "CD1",
        "CD2",
        "CE",
        "CE1",
        "CE2",
        "CE3",
        "CG",
        "CG1",
        "CG2",
        "CH2",
        "CZ",
        "CZ2",
        "CZ3",
        "N",
        "ND1",
        "ND2",
        "NE",
        "NE1",
        "NE2",
        "NH1",
        "NH2",
        "NZ",
        "O",
        "OD1",
        "OD2",
        "OE1",
        "OE2",
        "OG",
        "OG1",
        "OH",
        "OXT",
        "SD",
        "SG",
        "misc",
    ],
}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

lig_feature_dims = (
    list(
        map(
            len,
            [
                allowable_features["possible_atomic_num_list"],
                allowable_features["possible_chirality_list"],
                allowable_features["possible_degree_list"],
                allowable_features["possible_formal_charge_list"],
                allowable_features["possible_implicit_valence_list"],
                allowable_features["possible_numH_list"],
                allowable_features["possible_number_radical_e_list"],
                allowable_features["possible_hybridization_list"],
                allowable_features["possible_is_aromatic_list"],
                allowable_features["possible_numring_list"],
                allowable_features["possible_is_in_ring3_list"],
                allowable_features["possible_is_in_ring4_list"],
                allowable_features["possible_is_in_ring5_list"],
                allowable_features["possible_is_in_ring6_list"],
                allowable_features["possible_is_in_ring7_list"],
                allowable_features["possible_is_in_ring8_list"],
            ],
        )
    ),
    0,
)  # number of scalar features

rec_atom_feature_dims = (
    list(
        map(
            len,
            [
                allowable_features["possible_amino_acids"],
                allowable_features["possible_atomic_num_list"],
                allowable_features["possible_atom_type_2"],
                allowable_features["possible_atom_type_3"],
            ],
        )
    ),
    0,
)

rec_residue_feature_dims = (
    list(map(len, [allowable_features["possible_amino_acids"]])),
    0,
)


def lig_atom_featurizer(mol):
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append(
            [
                safe_index(
                    allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()
                ),
                allowable_features["possible_chirality_list"].index(
                    str(atom.GetChiralTag())
                ),
                safe_index(
                    allowable_features["possible_degree_list"], atom.GetTotalDegree()
                ),
                safe_index(
                    allowable_features["possible_formal_charge_list"],
                    atom.GetFormalCharge(),
                ),
                safe_index(
                    allowable_features["possible_implicit_valence_list"],
                    atom.GetImplicitValence(),
                ),
                safe_index(
                    allowable_features["possible_numH_list"], atom.GetTotalNumHs()
                ),
                safe_index(
                    allowable_features["possible_number_radical_e_list"],
                    atom.GetNumRadicalElectrons(),
                ),
                safe_index(
                    allowable_features["possible_hybridization_list"],
                    str(atom.GetHybridization()),
                ),
                allowable_features["possible_is_aromatic_list"].index(
                    atom.GetIsAromatic()
                ),
                safe_index(
                    allowable_features["possible_numring_list"],
                    ringinfo.NumAtomRings(idx),
                ),
                allowable_features["possible_is_in_ring3_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 3)
                ),
                allowable_features["possible_is_in_ring4_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 4)
                ),
                allowable_features["possible_is_in_ring5_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 5)
                ),
                allowable_features["possible_is_in_ring6_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 6)
                ),
                allowable_features["possible_is_in_ring7_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 7)
                ),
                allowable_features["possible_is_in_ring8_list"].index(
                    ringinfo.IsAtomInRingOfSize(idx, 8)
                ),
            ]
        )

    return torch.tensor(atom_features_list)
    
def remove_all_hs(mol):
    params = AllChem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    params.removeDefiningBondStereo = True
    params.removeDegreeZero = True
    params.removeDummyNeighbors = True
    params.removeHigherDegrees = True
    params.removeHydrides = True
    params.removeInSGroups = True
    params.removeIsotopes = True
    params.removeMapped = True
    params.removeNonimplicit = True
    params.removeNontetrahedralNeighbors = True
    params.removeOnlyHNeighbors = True
    params.removeWithQuery = True
    params.removeWithWedgedBond = True
    return AllChem.RemoveHs(mol, params, sanitize=False)

def remove_conformer_Hs(mol, confs):
    atomic_nums = []
    for at in mol.GetAtoms():
        atomic_nums.append(at.GetAtomicNum())
    atomic_nums = np.array(atomic_nums)
    mol = remove_all_hs(mol)
    assert mol.GetNumAtoms() == (atomic_nums != 1).sum()
    return mol, [conf[atomic_nums != 1] for conf in confs]
    
    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    
    for j, coords in enumerate(confs):
        conf = AllChem.Conformer(mol.GetNumAtoms())
        for i in range(coords.shape[0]):
            conf.SetAtomPosition(i, Point3D(*coords[i]))
        mol.AddConformer(conf)

    mol = remove_all_hs(mol)
    pos = []
    for conf in mol.GetConformers():
        pos.append(conf.GetPositions())
    return mol, pos

def get_symmetry_rmsd(mol, coords1, coords2, mol2=None, minimize=False, removeHs=False):
    mol2 = mol2 or mol
    if removeHs:
        mol, coords1 = remove_conformer_Hs(mol, coords1)
        mol2, coords2 = remove_conformer_Hs(mol2, coords2)
    mol = molecule.Molecule.from_rdkit(mol)
    mol2 = molecule.Molecule.from_rdkit(mol2)
    RMSD = rmsd.symmrmsd(
        coords1[0],
        coords2,
        mol.atomicnums,
        mol2.atomicnums,
        mol.adjacency_matrix,
        mol2.adjacency_matrix,
        minimize=minimize
    )

    return RMSD
def generate_conformer(mol):
    mol = AllChem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    ps.useExpTorsionAnglePrefs = False
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        logger.warning('RDkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
    mol = AllChem.RemoveHs(mol)
    return mol.GetConformer(0)
    
def rec_residue_featurizer(rec):
    feature_list = []
    for residue in rec.get_residues():
        feature_list.append(
            [
                safe_index(
                    allowable_features["possible_amino_acids"], residue.get_resname()
                )
            ]
        )
    return torch.tensor(feature_list, dtype=torch.float32)  # (N_res, 1)


def safe_index(l, e):
    """Return index of element e in list l. If e is not present, return the last index"""
    try:
        return l.index(e)
    except:
        return len(l) - 1


def parse_receptor(pdbid, pdbbind_dir, esmfold=False):
    if esmfold:
        path = os.path.join(pdbbind_dir, pdbid, f"{pdbid}_protein_esmfold_aligned_tr.pdb")
    else:
        path = os.path.join(pdbbind_dir, pdbid, f"{pdbid}_protein_processed.pdb")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure("random_id", path)
        rec = structure[0]
    return rec


def extract_receptor_structure(rec, lm_embedding_chains=None):
    c_alpha_coords, n_coords, c_coords = [], [], []
    all_atom_coords = []
    valid_chain_ids = []
    lengths = []

    meta = {'valid_chains': 0, 'valid_residues': 0, 'invalid_residues': 0, 'seq': []}

    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords, chain_n_coords, chain_c_coords = [], [], []
        
        count = 0
        invalid_res_ids = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                invalid_res_ids.append(residue.get_id())
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))
            
            if c_alpha != None and n != None and c != None:
                # only append residue if it is an amino acid and not some weird molecule that is part of the complex
                chain_c_alpha_coords.append(c_alpha)
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_coords.append(np.array(residue_coords))
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())

        meta['valid_residues'] += count
        meta['invalid_residues'] += len(invalid_res_ids)
        
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)

        meta['seq'].append(str(Polypeptide(chain).get_sequence()))
        
        c_alpha_coords.append(np.array(chain_c_alpha_coords))
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        all_atom_coords.append(chain_coords)
        if not count == 0:
            valid_chain_ids.append(chain.get_id())
            meta['valid_chains'] += 1
        
    valid_c_alpha_coords, valid_n_coords,valid_c_coords = [], [], []
    valid_all_atom_coords = []
    invalid_chain_ids = []

    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_c_alpha_coords.append(c_alpha_coords[i])
            valid_n_coords.append(n_coords[i])
            valid_c_coords.append(c_coords[i])
            valid_all_atom_coords.append(all_atom_coords[i])

        else:
            invalid_chain_ids.append(chain.get_id())

    all_atom_coords = [item for sublist in valid_all_atom_coords for item in sublist] 
    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]
    
    for invalid_id in invalid_chain_ids:
        rec.detach_child(invalid_id)

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    return {
        'meta': meta,
        'biopython_structure': rec,
        'c_alpha_coords': c_alpha_coords,
        'all_atom_coords': all_atom_coords,
    }





def get_lig_graph(mol, complex_graph, get_masks=False): # , remove_hs=False, rdkit_confs=False):
    
    
    complex_graph["ligand"].pos = torch.tensor(
        mol.GetConformers()[0].GetPositions()
    ).float()
    
    atom_feats = lig_atom_featurizer(mol)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += (
            2 * [bonds[bond.GetBondType()]]
            if bond.GetBondType() != BT.UNSPECIFIED
            else [0, 0]
        )

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    complex_graph["ligand"].node_attr = atom_feats
    complex_graph["ligand", "lig_bond", "ligand"].edge_index = edge_index
    complex_graph["ligand", "lig_bond", "ligand"].edge_attr = edge_attr

    try:
        complex_graph['meta'] |= {
            'lig_atoms': mol.GetNumAtoms(),
            'smiles': Chem.MolToSmiles(mol),
            'lig_bonds': edge_attr.shape[0] // 2
        }
    except:
        pass
    
   
def get_calpha_graph(
    rec,
    complex_graph,
    cutoff=20,
    max_neighbor=None,
    all_atoms=False,
    atom_radius=5.,
    atom_max_neighbors=8,
):
    num_residues = len(rec['c_alpha_coords'])
    if num_residues <= 1:
        raise ValueError(f"rec contains only 1 residue!")

    # Build the k-NN graph
    distances = spa.distance.cdist(rec['c_alpha_coords'], rec['c_alpha_coords'])
    src_list = []
    dst_list = []
    # mean_norm_list = []
    for i in range(num_residues):
        dst = list(np.where(distances[i, :] < cutoff)[0])
        dst.remove(i)
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(distances[i, :]))[1 : max_neighbor + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[
                1:2
            ]  # choose second because first is i itself
            logger.warning(
                f"The c_alpha_cutoff {cutoff} was too small for one c_alpha such that it had no neighbors. "
                f"So we connected it to the closest other c_alpha"
            )
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)

    assert len(src_list) == len(dst_list)

    node_feat = rec_residue_featurizer(rec['biopython_structure'])
    complex_graph["receptor"].node_attr = node_feat
    complex_graph["receptor"].pos = torch.from_numpy(rec['c_alpha_coords']).float()
    complex_graph["receptor", "rec_contact", "receptor"].edge_index = torch.from_numpy(
        np.asarray([src_list, dst_list])
    )

    if all_atoms:


        src_c_alpha_idx = np.concatenate([np.asarray([i]*len(l)) for i, l in enumerate(rec['all_atom_coords'])])
        atom_feat = torch.from_numpy(np.asarray(rec_atom_featurizer(rec['biopython_structure'])))
        atom_coords = torch.from_numpy(np.concatenate(rec['all_atom_coords'], axis=0)).float()
    
        not_hs = (atom_feat[:, 1] != 0)
        src_c_alpha_idx = src_c_alpha_idx[not_hs]
        atom_feat = atom_feat[not_hs]
        atom_coords = atom_coords[not_hs]

        atoms_edge_index = radius_graph(atom_coords, atom_radius, max_num_neighbors=atom_max_neighbors if atom_max_neighbors else 1000)
        atom_res_edge_index = torch.from_numpy(np.asarray([np.arange(len(atom_feat)), src_c_alpha_idx])).long()
    
        complex_graph['atom'].node_attr = atom_feat
        complex_graph['atom'].pos = atom_coords
        complex_graph['atom', 'atom_contact', 'atom'].edge_index = atoms_edge_index
        complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index = atom_res_edge_index
    
    
    
def rec_atom_featurizer(rec):
    atom_feats = []
    atom_coords = []
    for i, atom in enumerate(rec.get_atoms()):
        atom_name, element = atom.name, atom.element
        if element == "CD":
            element = "C"
        assert not element == ""
        try:
            atomic_num = periodic_table.GetAtomicNumber(element)
        except:
            atomic_num = -1
        atom_feat = [
            safe_index(
                allowable_features["possible_amino_acids"],
                atom.get_parent().get_resname(),
            ),
            safe_index(allowable_features["possible_atomic_num_list"], atomic_num),
            safe_index(
                allowable_features["possible_atom_type_2"], (atom_name + "*")[:2]
            ),
            safe_index(allowable_features["possible_atom_type_3"], atom_name),
        ]
        atom_feats.append(atom_feat)
        atom_coords.append(list(atom.get_vector()))
    return atom_feats #, atom_coords


def write_mol_with_coords(mol, new_coords, path):
    w = Chem.SDWriter(path)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x, y, z = new_coords.astype(np.double)[i]
        conf.SetAtomPosition(i, Point3D(x, y, z))
    w.write(mol)
    w.close()

def read_mol(pdbbind_dir, name, remove_hs=False):
    lig = read_molecule(
        os.path.join(pdbbind_dir, name, f"{name}_ligand.mol2"),
        remove_hs=remove_hs,
        sanitize=True,
    )
    
    return lig

def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    if molecule_file.endswith(".mol2"):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith(".sdf"):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith(".pdbqt"):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ""
        for line in pdbqt_data:
            pdb_block += "{}\n".format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError(
            "Expect the format of the molecule_file to be "
            "one of .mol2, .sdf, .pdbqt and .pdb, got {}".format(molecule_file)
        )

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn("Unable to compute charges for the molecule.")

        if remove_hs:
            mol = RemoveHs(mol, sanitize=sanitize)
    except Exception as e:
        logger.warning(f"RDKit was unable to read the molecule: {e}")
        return None

    return mol


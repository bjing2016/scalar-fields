import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pdbbind_dir', default='data/PDBBind_processed')
parser.add_argument('--split_path', default='splits/timesplit_test')
parser.add_argument('--num_confs', type=int, default=32)
parser.add_argument('--num_rots', type=int, default=32)
parser.add_argument('--num_trans', type=int, default=32)
parser.add_argument('--torsion_sigma', type=float, default=1.67) # pi / 2
parser.add_argument('--rot_sigma', type=float, default=0.5)
parser.add_argument('--tr_sigma', type=float, default=1)
parser.add_argument('--num_workers', type=int, default=1)
args = parser.parse_args()

from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from datasets.process_mols import read_mol
import numpy as np
import networkx as nx
import copy, tqdm
import pandas as pd
from scipy.spatial.transform import Rotation as R
from spyrmsd import rmsd, molecule
from multiprocessing import Pool

def main():

    split = list(map(lambda x: x.strip(), open(args.split_path)))

    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map

    for _ in tqdm.tqdm(__map__(job, split), total=len(split)):
        pass
    
def job(name):
    try:
        mol = read_mol(args.pdbbind_dir, name, remove_hs=False)
        poses, meta, confs, rots, offsets = get_confs(mol)
        print(mol.GetNumAtoms(), meta['rmsd'][1:].min(), meta['rmsd'][1:].median())
        path = f"{args.pdbbind_dir}/{name}/{name}_poses.sdf"
        write_confs(mol, poses, path)
        np.savez(f"{args.pdbbind_dir}/{name}/{name}_poses.npz", 
                 poses=np.array(poses), confs=confs, rots=rots, offsets=offsets)
        meta.to_csv(f"{args.pdbbind_dir}/{name}/{name}_poses.csv")

    except Exception as e:
        print(name, e)
def write_confs(mol, confs, path):
    w = AllChem.SDWriter(path)
    conf = mol.GetConformer()
    for j in range(len(confs)):
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i,Point3D(*confs[j][i]))
        w.write(mol)
        
        
def get_confs(mol):

    row, col = [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
    edge_index = np.array([row, col]).T
    mask_edges, mask_rotate = get_transformation_mask(edge_index)
    pos = mol.GetConformer(0).GetPositions()
    confs = [pos - pos.mean(0)]

    for _ in range(31):
        new_pos = modify_conformer(pos, edge_index[mask_edges], 
                mask_rotate, args.torsion_sigma * np.random.rand(mask_edges.sum()))
        confs.append(new_pos - new_pos.mean(0))
    kabsch_rmsds = get_symmetry_rmsd(mol, [confs[0]], confs, minimize=False, removeHs=True)
    rots = []
    for _ in range(32):
        rot = np.random.randn(3) * args.rot_sigma
        rots.append(R.from_rotvec(rot).as_matrix())
    rots = np.array(rots)
    rots[0] = np.eye(3)
    
    offsets = np.random.randn(32, 3) * args.tr_sigma
    offsets[0] = 0
    meta = []
    all_pos = []
    
    for i in range(32):
        for j in range(32):
            for k in range(32):
                all_pos.append(confs[i] @ rots[j] + offsets[k] + pos.mean(0))
                meta.append({
                    'conf_idx': i,
                    'rot_idx': j,
                    'tr_idx': k,
                    'centroid_rmsd': np.linalg.norm(offsets[k]),
                    'rotation_angle': np.linalg.norm(R.from_matrix(rots[j]).as_rotvec()),
                    'kabsch_rmsd': kabsch_rmsds[i]
                })
    
    conf_rmsds = get_symmetry_rmsd(mol, [pos], all_pos, minimize=False, removeHs=True)
    meta = pd.DataFrame(meta)
    meta['rmsd'] = conf_rmsds
    return all_pos, meta, confs, rots, offsets

def remove_conformer_Hs(mol, confs):
    atomic_nums = []
    for at in mol.GetAtoms():
        atomic_nums.append(at.GetAtomicNum())
    atomic_nums = np.array(atomic_nums)
    mol = remove_all_hs(mol)
    assert mol.GetNumAtoms() == (atomic_nums != 1).sum()
    return mol, [conf[atomic_nums != 1] for conf in confs]

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
    return AllChem.RemoveHs(mol, params)

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
def get_transformation_mask(edges):
    G = nx.from_edgelist(edges)
    to_rotate = []
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i + 1, 1]
        assert edges[i + 1, 0] == edges[i, 1]

        G2 = copy.deepcopy(G)

        if len(list(nx.connected_components(G2))) != 1:
            return None, None

        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])
    
    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(edges.shape[0]):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate

def modify_conformer(pos, edge_index, mask_rotate, torsion_updates):
    pos = np.copy(pos)
    for idx_edge, e in enumerate(edge_index):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated

        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = (
            rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec)
        )  # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (
            pos[mask_rotate[idx_edge]] - pos[v]
        ) @ rot_mat.T + pos[v]

    return pos

if __name__ == '__main__':
    main()

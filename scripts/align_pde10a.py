from tqdm import tqdm
import numpy as np
from scipy.optimize import minimize, Bounds
from rdkit import Chem
import scipy.spatial as spa
from Bio.PDB import PDBParser
from rdkit.Geometry import Point3D
from biopandas.pdb import PandasPdb
from Bio.SeqUtils import seq1
from Bio import pairwise2
biopython_parser = PDBParser()

def align_prediction(smoothing_factor, pdbbind_calpha_coords, omegafold_calpha_coords, pdbbind_ligand_coords, return_rotation=False):
    pdbbind_dists = spa.distance.cdist(pdbbind_calpha_coords, pdbbind_ligand_coords)
    weights = np.exp(-1 * smoothing_factor * np.amin(pdbbind_dists, axis=1))
    
    pdbbind_calpha_centroid = np.sum(np.expand_dims(weights, axis=1) * pdbbind_calpha_coords, axis=0) / np.sum(weights)
    omegafold_calpha_centroid = np.sum(np.expand_dims(weights, axis=1) * omegafold_calpha_coords, axis=0) / np.sum(weights)
    centered_pdbbind_calpha_coords = pdbbind_calpha_coords - pdbbind_calpha_centroid
    centered_omegafold_calpha_coords = omegafold_calpha_coords - omegafold_calpha_centroid
    centered_pdbbind_ligand_coords = pdbbind_ligand_coords - pdbbind_calpha_centroid
    
    rotation, rec_weighted_rmsd = spa.transform.Rotation.align_vectors(centered_pdbbind_calpha_coords, centered_omegafold_calpha_coords, weights)
    if return_rotation:
        return rotation, pdbbind_calpha_centroid, omegafold_calpha_centroid
    
    aligned_omegafold_calpha_coords = rotation.apply(centered_omegafold_calpha_coords)
    aligned_omegafold_pdbbind_dists = spa.distance.cdist(aligned_omegafold_calpha_coords, centered_pdbbind_ligand_coords)
    inv_r_rmse = np.sqrt(np.mean(((1 / pdbbind_dists) - (1 / aligned_omegafold_pdbbind_dists)) ** 2))
    return inv_r_rmse



for pdb_id in tqdm(open('splits/pde10a')):
    pdb_id = pdb_id.strip()

    ref_ppdb = PandasPdb().read_pdb('data/pde10a/5sfs/5sfs_protein_processed.pdb')
    prot_ppdb = PandasPdb().read_pdb(f'data/pde10a/{pdb_id}/{pdb_id}_protein_processed.pdb')
    ligand = Chem.SDMolSupplier(f'data/pde10a/{pdb_id}/{pdb_id}_ligand.sdf', sanitize=False, removeHs=False)[0]
    print(f'data/pde10a/{pdb_id}/{pdb_id}_ligand.sdf', flush=True)
    try:
        ligand_coords = ligand.GetConformer().GetPositions()
    except:
        import pdb
        pdb.set_trace()

    ref_CA_mask = ref_ppdb.df['ATOM']['atom_name'] == 'CA'
    prot_CA_mask = prot_ppdb.df['ATOM']['atom_name'] == 'CA'

    ref_seq = ref_ppdb.df['ATOM'][ref_CA_mask]['residue_name']
    prot_seq = prot_ppdb.df['ATOM'][prot_CA_mask]['residue_name']

    ref_seq = seq1(''.join(ref_seq))
    prot_seq = seq1(''.join(prot_seq))

    alignment = pairwise2.align.globalxx(ref_seq, prot_seq)[0]
    maskA, maskB = [c != '-' for c in alignment.seqA], [c != '-' for c in alignment.seqB]

    ref_mask = (np.cumsum(maskA) - 1)[np.array(maskA) & np.array(maskB)]
    prot_mask = (np.cumsum(maskB) - 1)[np.array(maskA) & np.array(maskB)]

    ref_coords = np.array(ref_ppdb.df['ATOM'][ref_CA_mask].iloc[ref_mask][['x_coord', 'y_coord', 'z_coord']])
    prot_coords = np.array(prot_ppdb.df['ATOM'][prot_CA_mask].iloc[prot_mask][['x_coord', 'y_coord', 'z_coord']])
    
    res = minimize(
        align_prediction,
        [0.1],
        bounds=Bounds([0.0],[1.0]),
        args=(
            prot_coords, # ground truth goes here
            ref_coords, # substitution goes here
            ligand_coords
        ),
        tol=1e-8
    )

    smoothing_factor = res.x
    inv_r_rmse = res.fun
    rotation, pdbbind_calpha_centroid, omegafold_calpha_centroid = align_prediction(
        smoothing_factor,
        prot_coords, # pdbbind
        ref_coords, # omegafold
        ligand_coords,
        True
    )

    
    # ppdb_omegafold_aligned = rotation.apply(ppdb_omegafold_pre_rot - omegafold_calpha_centroid) + pdbbind_calpha_centroid
    # R(X - O) + P
    # R^T(X - P) + O

    new_lig_coords = (ligand.GetConformer().GetPositions() - pdbbind_calpha_centroid) @ rotation.as_matrix() + omegafold_calpha_centroid

    w = Chem.SDWriter(f'data/pde10a/{pdb_id}/{pdb_id}_ligand_aligned.sdf')
    conf = ligand.GetConformer()
    for i in range(ligand.GetNumAtoms()):
        conf.SetAtomPosition(i,Point3D(*new_lig_coords[i]))
    w.write(ligand)
import pandas as pd
import tqdm, os, time, argparse, subprocess
import numpy as np
from rdkit.Chem import AllChem
from spyrmsd import rmsd, molecule

parser = argparse.ArgumentParser()
parser.add_argument('--pdbbind_dir', default='data/PDBBind_processed')
parser.add_argument('--splits', default='splits/timesplit_test')
parser.add_argument('--poses_suffix', default='poses.sdf')
parser.add_argument('--receptor_suffix', default='protein_processed.pdb')
parser.add_argument('--outdir', default='workdir/gnina_default')
parser.add_argument('--outcsv', default='workdir/gnina_default.csv')
parser.add_argument('--scoring', choices=['ad4_scoring', 'vina', 'gnina', 'vinardo'], default='vina')
args = parser.parse_args()

my_pid = os.getpid()



def main():

    pdb_ids = open(args.splits).read().split('\n')[:-1]
    os.makedirs(args.outdir, exist_ok=True)
    
    entries = []
    
    for name in tqdm.tqdm(pdb_ids):
         
        r_path = f"{args.pdbbind_dir}/{name}/{name}_{args.receptor_suffix}"
        l_path = f"{args.pdbbind_dir}/{name}/{name}_{args.poses_suffix}"
        out_path = f"{args.outdir}/{name}.out"
        cmd = ['gnina', '--score_only', '-r', r_path, '-l', l_path]
        if args.scoring != 'gnina':
            cmd += ['--scoring', args.scoring, '--cnn_scoring', 'none']
        
        print(' '.join(cmd))
        
        start = time.time()
        with open(out_path, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)
        dur = time.time() - start
            
        entries.append({'name': name, 'time': dur})
        print(entries[-1])
        
    pd.DataFrame(entries).to_csv(args.outcsv)
    
            
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
    return AllChem.RemoveHs(mol, params, sanitize=False)

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

main()
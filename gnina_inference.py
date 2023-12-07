import pandas as pd
import tqdm, os, time, argparse, subprocess
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from spyrmsd import rmsd, molecule

parser = argparse.ArgumentParser()
parser.add_argument('--pdbbind_dir', default='../data/PDBBind_processed')
parser.add_argument('--splits', default='splits/timesplit_test')
parser.add_argument('--ligand_suffix', default='ligand_rigid.pdbqt')
parser.add_argument('--receptor_suffix', default='protein_processed.pdb')
parser.add_argument('--common_receptor', default=None)
parser.add_argument('--autobox_suffix', default='ligand.mol2')
parser.add_argument('--common_autobox', default=None)
parser.add_argument('--ref_suffix', default='ligand.mol2')
parser.add_argument('--max_mc_steps', default=None)
parser.add_argument('--minimize_iters', default=None)
parser.add_argument('--score_only', action='store_true')
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
         
        r_path = args.common_receptor or f"{args.pdbbind_dir}/{name}/{name}_{args.receptor_suffix}"
        l_path = f"{args.pdbbind_dir}/{name}/{name}_{args.ligand_suffix}"
        box_path = args.common_autobox or f"{args.pdbbind_dir}/{name}/{name}_{args.autobox_suffix}"
        ref_path = f"{args.pdbbind_dir}/{name}/{name}_{args.ref_suffix}"
        out_path = f"{args.outdir}/{name}_out.sdf"
        cmd = ['gnina', '-r', r_path, '-l', l_path, '--autobox_ligand', box_path, '-o', out_path]
        if args.scoring != 'gnina':
            cmd += ['--scoring', args.scoring, '--cnn_scoring', 'none']
        if args.max_mc_steps: cmd += ['--max_mc_steps', args.max_mc_steps]
        if args.minimize_iters: cmd += ['--minimize_iters', args.minimize_iters]
        if args.score_only: cmd += ['--score_only']

        print(' '.join(cmd))
        start = time.time()
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dur = time.time() - start
            
        subprocess.run(['obabel', ref_path, '-O',  f"/tmp/{my_pid}.mol2"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        try:
            ref_mol = Chem.MolFromMol2File(f'/tmp/{my_pid}.mol2', sanitize=False, removeHs=False)
            docked_mol = Chem.SDMolSupplier(out_path, sanitize=False, removeHs=False)[0]
            rmsd = get_symmetry_rmsd(
                ref_mol, [ref_mol.GetConformer(0).GetPositions()], 
                [docked_mol.GetConformer(0).GetPositions()], docked_mol, 
            removeHs=True)
            entries.append({'name': name, 'rmsd': rmsd[0], 'time': dur})
            print(entries[-1])
            
        except Exception as e:
            print("Error", e)
            entries.append({'name': name, 'rmsd': np.nan, 'time': dur})

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
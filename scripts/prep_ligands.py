import os, tqdm, sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from scipy.spatial.transform import Rotation as R

if sys.argv[1] == 'pdbbind':
    split_path = 'splits/timesplit_test'
elif sys.argv[1] == 'pde10a':
    split_path = 'splits/pde10a'

for pdb_id in tqdm.tqdm(open(split_path)):
    pdb_id = pdb_id.strip()
    if sys.argv[1] == 'pdbbind':
        mol = Chem.MolFromMol2File(f"data/PDBBind_processed/{pdb_id}/{pdb_id}_ligand.mol2", sanitize=False, removeHs=False)
    elif sys.argv[1] == 'pde10a':
        mol = Chem.SDMolSupplier(f"data/pde10a/{pdb_id}/{pdb_id}_ligand.sdf", sanitize=False)[0]
    
    coords = mol.GetConformer(0).GetPositions()
    coords = (coords - coords.mean(0)) @ R.random().as_matrix().T
    for i in range(coords.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, Point3D(*coords[i]))
    AllChem.SDWriter('/tmp/tmp.sdf').write(mol)
    if sys.argv[1] == 'pdbbind':
        out_path = f"data/PDBBind_processed/{pdb_id}/{pdb_id}_ligand_rigid.pdbqt"
    elif sys.argv[1] == 'pde10a':
        out_path = f"data/pde10a/{pdb_id}/{pdb_id}_ligand_rigid.pdbqt"
    
    cmd = f'obabel /tmp/tmp.sdf -O {out_path} -p 7.4 -xh'
    os.system(cmd)
    with open(out_path) as f:
        pdbqt = f.read().split('\n')[:-2]
    pdbqt = [line for line in pdbqt if 'BRANCH' not in line and 'ENDROOT' not in line] + ['ENDROOT', 'TORSDOF 0']
    with open(out_path, 'w') as f:
        f.write('\n'.join(pdbqt))
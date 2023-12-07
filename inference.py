import argparse
parser = argparse.ArgumentParser()

### Args needed by dataset
parser.add_argument("--pdbbind_dir", type=str, default="data/PDBBind_processed")
parser.add_argument("--test_split_path", type=str, default="splits/timesplit_test")
parser.add_argument("--receptor_radius", type=float, default=30)
parser.add_argument("--c_alpha_max_neighbors", type=int, default=10)
parser.add_argument('--atom_radius', type=float, default=5)
parser.add_argument('--atom_max_neighbors', type=int, default=8)

### Inference args
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--outjson", type=str, default='workdir/default.json')
parser.add_argument("--esmfold", action='store_true')
parser.add_argument("--ablate_chemistry", action='store_true')
parser.add_argument("--outdir", default='workdir/outdir_default')
parser.add_argument("--mode", choices=['R', 'T'], required=True)

## R-mode settings
parser.add_argument("--box_radius", type=int, default=4)
parser.add_argument("--box_grid_count", type=int, default=9)
parser.add_argument("--fft_lmax", type=int, default=50)

## T-mode settings
parser.add_argument("--so3_grid_resolution", type=int, default=2)
parser.add_argument("--fft_scaling", type=int, default=1)


args = parser.parse_args()
args.overfit = False
args.max_lig_size = float('nan')
args.max_protein_len = float('nan')

import scipy.stats
import tqdm, torch, os, time, json
from datasets.pdbbind import PDBBind
from model.wrapper import ModelWrapper
import numpy as np
from datasets.process_mols import get_symmetry_rmsd
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from utils import so3fft, fft, so3_grid
from scipy.spatial.transform import Rotation

class CudaTimer:
    def __init__(self):
        self.event = torch.cuda.Event(enable_timing=True)
        self.event.record()
    def tick(self):
        now = torch.cuda.Event(enable_timing=True)
        now.record()
        torch.cuda.synchronize()
        time = self.event.elapsed_time(now)
        self.event = now
        return time

rots = None # hacky way to only compute these once
@torch.no_grad()
def main():

    model = ModelWrapper.load_from_checkpoint(args.ckpt)
    args.all_atoms = model.args.all_atoms
    args.ablate_chemistry = args.ablate_chemistry or model.args.ablate_chemistry
    os.makedirs(args.outdir, exist_ok=True)
    
    test_dataset = PDBBind(args, split_path=args.test_split_path, return_all=True, esmfold=args.esmfold)
    
    rmsds = {}
    timings = {}
    for i in tqdm.trange(len(test_dataset)):
        try:
            batch = test_dataset.get(i)
            batch.pocket_center = batch['ligand'].pos.mean(0)
            
            #### Erase the true pose
            random_rot = torch.from_numpy(Rotation.random().as_matrix()).float()
            batch['ligand'].pos = (batch['ligand'].pos - batch.pocket_center) @ random_rot.T
            if args.mode == 'R':
                grid_spacing = 2 * args.box_radius / (args.box_grid_count - 1)
            elif args.mode == 'T':
                grid_spacing = model.args.fft_resolution / args.fft_scaling
            grid_offset = grid_spacing * (torch.rand(3) - 0.5)
            batch.pocket_center += grid_offset
                
            batch.cuda()
            if args.mode == 'R':
                rots, out_pos, timing = do_batch_R(batch, model)
            elif args.mode == 'T':
                rots, out_pos, timing = do_batch_T(batch, model)
            out_pos = out_pos.cpu().numpy()
            
            ## Compute RMSD
            true_pos = batch.rdkit_mol.GetConformer(0).GetPositions() 
            rmsd = get_symmetry_rmsd(batch.rdkit_mol, [true_pos], [out_pos], removeHs=True)[0]
            mol = batch.rdkit_mol
            conf = mol.GetConformer()
            for i in range(out_pos.shape[0]):
                x, y, z = out_pos[i]
                conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
            AllChem.SDWriter(f"{args.outdir}/{batch.name}.sdf").write(mol)

            possible_rots = torch.einsum('bmn,pn->bpm', rots, batch['ligand'].pos)
            true_pos_centered = torch.from_numpy(true_pos - true_pos.mean(0)).to(possible_rots)
            rot_offset = torch.square(possible_rots - true_pos_centered).sum(-1).mean(-1)
            rmsds[batch.name] = {
                'rmsd': rmsd,
                'tr_rmsd': np.linalg.norm(out_pos.mean(0) - true_pos.mean(0)),
                'tr_grid_offset': torch.norm(grid_offset).item(),
                'rot_grid_offset': torch.sqrt(rot_offset).min().item()
            }
            print(rmsds[batch.name])
            timings[batch.name] = timing
            print(timings[batch.name])
            
        except Exception as e:
            #raise e
            print('Error', batch.name, e)

    with open(args.outjson, 'w') as f:
        json.dump({'rmsds': rmsds, 'timings': timings}, f, indent=4)

def do_batch_R(batch, model):
    device = batch["ligand"].pos.device
    cache = model.cache
    if cache.cached_C is None:
        cache.to(device)
        cache.precompute_offsets()
    timing = {}
    master_start = time.time()
    timer = CudaTimer()
    #####################
    offsets = torch.linspace(-args.box_radius, args.box_radius, args.box_grid_count, device='cuda')
    offsets = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
    offsets = torch.stack(offsets, -1).reshape(-1, 3)
    timing['preprocessing'] = timer.tick()
    #######################
    prot_out = model.protein_model(batch, key="receptor", radius=False,
                    all_atoms=args.all_atoms)
    timing['protein_model'] = timer.tick()
    #######################
    lig_pos = batch['ligand'].pos
    ligand_out = model.ligand_model(batch, key="ligand", radius=True)
    timing['ligand_model'] = timer.tick()
    ########################
    lig_mask = lig_pos.norm(dim=-1) < model.args.so3_radius
    lig_anlm = cache.local_to_global(lig_pos[lig_mask].unsqueeze(-2), ligand_out[lig_mask]).sum(0)
    timing['ligand_coeffs'] = timer.tick()
    #########################
    prot_anlms = []
    for offset in tqdm.tqdm(offsets):
        prot_pos = batch['receptor'].pos - batch.pocket_center - offset
        prot_mask = prot_pos.norm(dim=-1) < model.args.so3_radius
        prot_anlm = cache.local_to_global(prot_pos[prot_mask].unsqueeze(-2), prot_out[prot_mask]).sum(0)
        prot_anlms.append(prot_anlm)

    prot_anlms = torch.stack(prot_anlms)
    timing['protein_coeffs'] = timer.tick()
    #########################
    field_cache = []
    for prot_anlm in torch.split(prot_anlms, 32):
        field = so3fft.so3_fft(prot_anlm, lig_anlm, cache.global_irreps, cache.global_rbf_I / 100, lmax=args.fft_lmax).sum(-4)
        field_cache.append(field)
    field = torch.cat(field_cache)
    timing['fft'] = timer.tick()
    ##########################
    sorter = field.view(offsets.shape[0], -1).max(-1)[0].argsort()
    max_offset = offsets[sorter[-1]]
    max_field = field[sorter[-1]]
    a, b, c = so3fft.unindex_field(max_field, max_field.argmax().item())
    unrot = so3fft.wigner_D(1, a, b, c, real=True, order="YZX").to(device)
    out_pos = lig_pos @ unrot.T + batch.pocket_center + max_offset
    timing['readout'] = timer.tick()
    ##########################
    timing['all'] = 1000 * (time.time() - master_start)
    
    global rots
    if rots is None:
        idx = torch.arange(field.shape[-1], device='cuda')
        xi_idx, eta_idx, om_idx = torch.meshgrid(idx, idx, idx, indexing='ij')
        phi, theta, psi = so3fft.unindex_field(field, xi_idx=xi_idx, eta_idx=eta_idx, om_idx=om_idx)
        rots = so3fft.wigner_D(1, phi.flatten(), theta.flatten(), psi.flatten(), real=True, order='YZX')
    return rots, out_pos, timing


def do_batch_T(batch, model):
    name = batch.name
    device = batch['ligand'].pos.device
    
    cache = model.tr_cache
    if not cache.initialized:
        cache.initialize(
            R_grid_spacing=model.args.fft_resolution, 
            R_grid_diameter=model.args.box_diameter,
        device=device)
    timing = {}
    master_start = time.time()
    timer = CudaTimer()
    ### Preprocessing
    rots = so3_grid.grid_SO3(args.so3_grid_resolution)
    rots = Rotation.from_quat(rots).as_matrix()
    rots = torch.from_numpy(rots).float()
    Ds = model.ligand_model.sh_irreps.D_from_matrix(rots).to(device).float()
    rots = rots.to(device).float()
    timing['preprocessing'] = timer.tick()
    ######### Protein model
    prot_out = model.protein_model(batch, key="receptor", radius=False, all_atoms=model.args.all_atoms)
    timing['protein_model'] = timer.tick()
    ##### Protein coeffs
    prot_pos = batch['receptor'].pos - batch.pocket_center
    prot_mask = (prot_pos.abs() < model.args.box_diameter // 2).sum(1) == 3
    prot_fft = cache.render_fft(prot_pos[prot_mask], prot_out[prot_mask])  
    timing['protein_fft'] = timer.tick()
    ###### Ligand model
    ligand_out = model.ligand_model(batch, key="ligand", radius=True)
    timing['ligand_model'] = timer.tick()
    ####### Ligand coeffs
    lig_pos = batch['ligand'].pos
    lig_mask = lig_pos.norm(dim=-1) < model.args.box_diameter / 2
    fft_cache = []
    for j in range(rots.shape[0]):
        R = rots[j] 
        D = Ds[j]
        fft_cache.append(cache.render_fft(lig_pos[lig_mask] @ R.T, ligand_out[lig_mask] @ D.T))
    timing['ligand_fft'] = timer.tick()
    #################
    field_cache = []
    for j in range(rots.shape[0]):
        field_cache.append(fft.cross_correlate(cache, protein=prot_fft, 
                ligand=fft_cache[j], scaling=args.fft_scaling, sum=-4))
    del fft_cache
    field = torch.stack(field_cache)
    timing['cross_correlate'] = timer.tick()
    ##################### Restricting to the pocket
    del field_cache
    NR_grid = args.fft_scaling * cache.NR_grid 
    r_grid = torch.linspace(-cache.R_grid_diameter / 2, cache.R_grid_diameter / 2, NR_grid+1, device=device)[:NR_grid]
    rx, ry, rz = torch.meshgrid(r_grid, r_grid, r_grid, indexing="ij")
    rxyz = torch.stack([rx, ry, rz], -1)
    box_mask = (rxyz.abs() <= 4).sum(-1) == 3
    field = torch.where(box_mask, field, field.min())

    sorter = field.view(rots.shape[0], -1).max(-1)[0].argsort()
    max_rot = rots[sorter[-1]]
    max_field = field[sorter[-1]]
    xyz = rxyz.view(-1, 3)[max_field.argmax().item()]
    out_pos = lig_pos @ max_rot.T + xyz + batch.pocket_center
    timing['readout'] = timer.tick()
    ########################
    timing['all'] = 1000 * (time.time() - master_start)
    return rots, out_pos, timing

if __name__ == "__main__":
    main()

import argparse
parser = argparse.ArgumentParser()

### Args needed by dataset
parser.add_argument("--receptor_radius", type=float, default=30)
parser.add_argument("--c_alpha_max_neighbors", type=int, default=10)
parser.add_argument('--atom_radius', type=float, default=5)
parser.add_argument('--atom_max_neighbors', type=int, default=8)

### Inference args
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--outjson", type=str, default='workdir/default.json')
parser.add_argument("--mode", choices=['R', 'T'], required=True)
parser.add_argument("--outdir", default='workdir/outdir_default')
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

import tqdm, torch, os, time, json
import numpy as np
from model.wrapper import ModelWrapper
from datasets import process_mols
from torch_geometric.data import HeteroData, Batch
from datasets.process_mols import get_symmetry_rmsd
from rdkit import Chem
from rdkit.Geometry import Point3D
from utils import so3fft, fft, so3_grid
from scipy.spatial.transform import Rotation
from torch_geometric.utils import unbatch

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


@torch.no_grad()
def main():
    pdb_ids = [pdb_id.strip() for pdb_id in open('splits/pde10a')]
    os.makedirs(args.outdir, exist_ok=True)
    
    model = ModelWrapper.load_from_checkpoint(args.ckpt)
    args.all_atoms = model.args.all_atoms
    model.cache.to('cuda')
    model.cache.precompute_offsets()
    model.tr_cache.initialize(
            R_grid_spacing=model.args.fft_resolution, 
            R_grid_diameter=model.args.box_diameter,
        device='cuda')

    timings = {'preprocess': {}}
    start = time.time()
    receptor = process_mols.parse_receptor('5sfs', 'data/pde10a')
    receptor = process_mols.extract_receptor_structure(receptor, lm_embedding_chains=None)
    
    ref_ligand = Chem.SDMolSupplier("data/pde10a/5sfs/5sfs_ligand.sdf")[0]
    
    complex_graph = HeteroData()
    complex_graph.pocket_center = torch.from_numpy(ref_ligand.GetConformer().GetPositions().mean(0)).float()

    ## Erase the exact pocket center
    if args.mode == 'R':
        grid_spacing = 2 * args.box_radius / (args.box_grid_count - 1)
    elif args.mode == 'T':
        grid_spacing = model.args.fft_resolution / args.fft_scaling
    complex_graph.pocket_center += grid_spacing * (torch.rand(3) - 0.5)
    
    process_mols.get_calpha_graph(receptor, complex_graph,
            cutoff=args.receptor_radius, max_neighbor=args.c_alpha_max_neighbors, all_atoms=args.all_atoms,
            atom_radius=args.atom_radius, atom_max_neighbors=args.atom_max_neighbors)
    complex_graph.cuda()
    
    ligands = {}
    ligand_graphs = {}
    for pdb_id in pdb_ids:
        ligands[pdb_id] = Chem.SDMolSupplier(f"data/pde10a/{pdb_id}/{pdb_id}_ligand_aligned.sdf", sanitize=True)[0]
        if ligands[pdb_id] is None: 
            print('RDKit could not read', pdb_id)
            continue
        ligand_graphs[pdb_id] = HeteroData()
        process_mols.get_lig_graph(ligands[pdb_id], ligand_graphs[pdb_id], get_masks=False)
        ### Erase ligand pose
        lig_pos = ligand_graphs[pdb_id]['ligand'].pos 
        lig_pos = (lig_pos - lig_pos.mean(0)) @ torch.from_numpy(Rotation.random().as_matrix()).float()
        ligand_graphs[pdb_id]['ligand'].pos = lig_pos

    ligand_batch_graph = Batch.from_data_list([ligand_graphs[pdb_id] for pdb_id in pdb_ids if ligands[pdb_id]])
    ligand_batch_graph.cuda()

    lig_pos = {}
    lig_pos_ = unbatch(ligand_batch_graph['ligand'].pos, ligand_batch_graph['ligand'].batch) 
    for pdb_id in pdb_ids:
        if not ligands[pdb_id]: continue
        lig_pos[pdb_id] = lig_pos_[len(lig_pos)]
    timings['preprocess']['loading'] = 1000*(time.time() - start)
    ##################################
    timer = CudaTimer()
    prot_out = model.protein_model(complex_graph, key="receptor", radius=False, all_atoms=args.all_atoms)
    timings['preprocess']['protein_model'] = timer.tick()
    ####################################
    ligand_out = model.ligand_model(ligand_batch_graph, key="ligand", radius=True)
    ligand_out_ = unbatch(ligand_out, ligand_batch_graph['ligand'].batch)    
    ligand_out = {}
    for pdb_id in pdb_ids:
        if not ligands[pdb_id]: continue
        ligand_out[pdb_id] = ligand_out_[len(ligand_out)]
    timings['preprocess']['ligand_model'] = timer.tick()
    #################################
    offsets, rots, out_pos = {'R': do_R, 'T': do_T}[args.mode](model, complex_graph, prot_out, pdb_ids, ligands, lig_pos, ligand_out, timer, timings)
    ################################
    rmsds = {}
    for pdb_id in pdb_ids:
        if not ligands[pdb_id]: continue
            
        true_pos = ligands[pdb_id].GetConformer().GetPositions()

        coords = out_pos[pdb_id].cpu().numpy()
        rmsd = get_symmetry_rmsd(ligands[pdb_id], [true_pos], [coords], removeHs=True)[0]

        mol = ligands[pdb_id]
        conf = mol.GetConformer()
        for i in range(coords.shape[0]):
            x, y, z = coords[i]
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

        Chem.SDWriter(f"{args.outdir}/{pdb_id}.sdf").write(mol)
        tr_offset = complex_graph.pocket_center + offsets - torch.from_numpy(true_pos.mean(0)).to(offsets)
        possible_rots = torch.einsum('bmn,pn->bpm', rots, lig_pos[pdb_id])
        true_pos_centered = torch.from_numpy(true_pos - true_pos.mean(0)).to(possible_rots)
        rot_offset = torch.square(possible_rots - true_pos_centered).sum(-1).mean(-1)
        rmsds[pdb_id] = {
            'rmsd': rmsd,
            'tr_rmsd': np.linalg.norm(coords.mean(0) - true_pos.mean(0)),
            'tr_grid_offset': torch.linalg.norm(tr_offset, dim=-1).min().item(),
            'rot_grid_offset': torch.sqrt(rot_offset).min().item()
        }
        print(rmsds[pdb_id])
    
    with open(args.outjson, 'w') as f:
        json.dump({'rmsds': rmsds, 'timings': timings}, f, indent=4)

def do_R(model, complex_graph, prot_out, pdb_ids, ligands, lig_pos, ligand_out, timer, timings):
    
    offsets = torch.linspace(-args.box_radius, args.box_radius, args.box_grid_count, device='cuda')
    offsets = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
    offsets = torch.stack(offsets, -1).reshape(-1, 3)
    cache = model.cache
    timings['preprocess']['offsets'] = timer.tick()
    ##################
    prot_anlms = []
    for offset in tqdm.tqdm(offsets):
        prot_pos = complex_graph['receptor'].pos - complex_graph.pocket_center - offset
        prot_mask = prot_pos.norm(dim=-1) < model.args.so3_radius
        prot_anlm = cache.local_to_global(prot_pos[prot_mask].unsqueeze(-2), prot_out[prot_mask]).sum(0)
        prot_anlms.append(prot_anlm)
    prot_anlms = torch.stack(prot_anlms)
    timings['preprocess']['protein_coeffs'] = timer.tick()
    ################################
    lig_anlms = {}
    for pdb_id in pdb_ids:
        if not ligands[pdb_id]: continue
        lig_mask = lig_pos[pdb_id].norm(dim=-1) < model.args.so3_radius
        lig_anlm = cache.local_to_global(lig_pos[pdb_id][lig_mask].unsqueeze(-2), ligand_out[pdb_id][lig_mask]).sum(0)
        lig_anlms[pdb_id] = lig_anlm
    timings['preprocess']['ligand_coeffs'] = timer.tick()
    ################################
    out_pos = {}
    
    for pdb_id in tqdm.tqdm(pdb_ids):
        if not ligands[pdb_id]: continue
        timings[pdb_id] = {}
        field_cache = []
        for prot_anlm in torch.split(prot_anlms, 32):
            field = so3fft.so3_fft(prot_anlm, lig_anlms[pdb_id], cache.global_irreps, cache.global_rbf_I / 100, lmax=args.fft_lmax).sum(-4)
            field_cache.append(field)
        field = torch.cat(field_cache)
        timings[pdb_id]['cross_correlate'] = timer.tick()
        ################ 
        sorter = field.view(offsets.shape[0], -1).max(-1)[0].argsort()
        max_field = field[sorter[-1]]
        a, b, c = so3fft.unindex_field(max_field, max_field.argmax().item())
        unrot = so3fft.wigner_D(1, a, b, c, real=True, order="YZX").to('cuda')
        out_pos[pdb_id] = lig_pos[pdb_id] @ unrot.T + complex_graph.pocket_center + offsets[sorter[-1]]
        timings[pdb_id]['readout'] = timer.tick()

    idx = torch.arange(field.shape[-1], device='cuda')
    xi_idx, eta_idx, om_idx = torch.meshgrid(idx, idx, idx, indexing='ij')
    phi, theta, psi = so3fft.unindex_field(field, xi_idx=xi_idx, eta_idx=eta_idx, om_idx=om_idx)
    rots = so3fft.wigner_D(1, phi.flatten(), theta.flatten(), psi.flatten(), real=True, order='YZX')
    return offsets, rots, out_pos
    
def do_T(model, complex_graph, prot_out, pdb_ids, ligands, lig_pos, ligand_out, timer, timings):

    ### Preprocessing
    rots = so3_grid.grid_SO3(args.so3_grid_resolution)
    rots = Rotation.from_quat(rots).as_matrix()
    rots = torch.from_numpy(rots).float()
    Ds = model.ligand_model.sh_irreps.D_from_matrix(rots).to('cuda').float()
    rots = rots.to('cuda').float()
    cache = model.tr_cache
    
    NR_grid = args.fft_scaling * cache.NR_grid 
    r_grid = torch.linspace(-cache.R_grid_diameter / 2, cache.R_grid_diameter / 2, NR_grid+1, device='cuda')[:NR_grid]
    rx, ry, rz = torch.meshgrid(r_grid, r_grid, r_grid, indexing="ij")
    rxyz = torch.stack([rx, ry, rz], -1)
    box_mask = (rxyz.abs() <= 4).sum(-1) == 3
    del r_grid, rx, ry, rz
    timings['preprocess']['grids'] = timer.tick()
    ############################################
    prot_pos = complex_graph['receptor'].pos - complex_graph.pocket_center 
    prot_mask = (prot_pos.abs() < model.args.box_diameter // 2).sum(1) == 3
    prot_fft = cache.render_fft(prot_pos[prot_mask], prot_out[prot_mask])  
    timings['preprocess']['protein_fft'] = timer.tick()
    ############################################
    out_pos = {}
    for pdb_id in tqdm.tqdm(pdb_ids):
        if not ligands[pdb_id]: continue
        timings[pdb_id] = {}
        lig_mask = (lig_pos[pdb_id].abs() < model.args.box_diameter // 2).sum(1) == 3
        fft_cache = []
        for j in range(rots.shape[0]):
            R = rots[j] 
            D = Ds[j]
            fft_cache.append(cache.render_fft(lig_pos[pdb_id][lig_mask] @ R.T, ligand_out[pdb_id][lig_mask] @ D.T))
        timings[pdb_id]['ligand_fft'] = timer.tick()
        ################
        field_cache = []
        for j in range(rots.shape[0]):
            field_cache.append(fft.cross_correlate(cache, protein=prot_fft, ligand=fft_cache[j], scaling=args.fft_scaling, sum=-4))
        del fft_cache
        
        timings[pdb_id]['cross_correlate'] = timer.tick()
        ################
        field_max = []
        for field in field_cache:
            field = torch.where(box_mask, field, field.min())
            field_max.append(field.max().item())
        sorter = np.array(field_max).argsort()
        max_rot = rots[sorter[-1]]
        max_field = field_cache[sorter[-1]]
        xyz = rxyz.view(-1, 3)[max_field.argmax().item()]
        out_pos[pdb_id] = lig_pos[pdb_id] @ max_rot.T + xyz + complex_graph.pocket_center
        timings[pdb_id]['readout'] = timer.tick()
        del field_cache

    return rxyz[box_mask], rots, out_pos
if __name__ == "__main__":
    main()

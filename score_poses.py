import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pdbbind_dir", type=str, default="data/PDBBind_processed")
parser.add_argument("--test_split_path", type=str, default="splits/timesplit_test")
parser.add_argument("--receptor_radius", type=float, default=30)
parser.add_argument("--c_alpha_max_neighbors", type=int, default=10)
parser.add_argument('--atom_radius', type=float, default=5)
parser.add_argument('--atom_max_neighbors', type=int, default=8)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--outdir", type=str, default='workdir/default')
parser.add_argument("--esmfold", action='store_true')
parser.add_argument("--mode", choices=['R', 'T'], default='R')
args = parser.parse_args()
args.overfit = False
args.max_lig_size = float('nan')
args.max_protein_len = float('nan')

import tqdm, torch, os, time
import numpy as np
from datasets.pdbbind import PDBBind
from model.wrapper import ModelWrapper
from torch_geometric.data import Batch
import pandas as pd
from torch_scatter import scatter_sum
from scipy.spatial.transform import Rotation
from utils import so3fft
from utils.logging import get_logger
logger = get_logger(__name__)


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

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
    
    model = ModelWrapper.load_from_checkpoint(args.ckpt)
    args.all_atoms = model.args.all_atoms
    test_dataset = PDBBind(args, split_path=args.test_split_path,
            return_all=True, esmfold=args.esmfold)
    
    model.cache.to('cuda')
    model.cache.precompute_offsets()
    timings = []
    for i in tqdm.trange(len(test_dataset)):
        try:
            batch = test_dataset.get(i)
            batch.cuda()
            if args.mode == 'R':
                timing = do_batch_R(batch, model)
            elif args.mode == 'T':
                timing = do_batch_T(batch, model)
            print(timing)
            timings.append(timing)
        except Exception as e:
            logger.warning(f'Error {batch.name} {e}')

    pd.DataFrame(timings).to_csv(os.path.join(args.outdir, 'timings.csv'))
            
def do_batch_R(batch, model):
    name = batch.name
    device = batch['ligand'].pos.device
    ####### Preprocessing which could have been avoided 
    npz = dict(np.load(f"{args.pdbbind_dir}/{name}/{name}_poses.npz"))
    withH_pos = batch['ligand'].pos.cpu()
    noH_pos = torch.from_numpy(npz['poses'][0])
    distmat = (withH_pos.unsqueeze(-2) - noH_pos.unsqueeze(-3)).norm(dim=-1)
    noH_idx = distmat.min(1)[1]
    num_atoms = batch['ligand'].pos.shape[0]
    num_confs = npz['confs'].shape[0]
    
    pocket_center = batch.full_rdkit_mol.GetConformer(0).GetPositions().mean(0)
    pocket_center = torch.from_numpy(pocket_center).to(device).float()
    ################
    timing = {'name': name}
    master_start = time.time()
    timer = CudaTimer()
    #####################
    big_batch = Batch.from_data_list([batch] * num_confs)    
    big_batch['ligand'].pos = lig_pos = torch.from_numpy(npz['confs'][:,noH_idx].reshape(-1, 3)).float().to(device)

    offsets = torch.from_numpy(npz['offsets']).float().to(device)
    rots = Rotation.from_matrix(npz['rots']).as_euler('zyz')
    rots = torch.from_numpy(rots).flip(-1).float().to(device)
    wigner_D = so3fft.irrep_wigner_D(model.cache.global_irreps, *rots.T)    
    timing['preprocessing'] = timer.tick()
    #######################################
    ligand_out = model.ligand_model(big_batch, key="ligand", radius=True)
    timing['ligand_model'] = timer.tick()
    #######################################
    lig_mask = lig_pos.norm(dim=-1) < model.args.so3_radius
    lig_anlm = model.cache.local_to_global(lig_pos[lig_mask].unsqueeze(-2), ligand_out[lig_mask])
    lig_anlm = scatter_sum(lig_anlm, big_batch['ligand'].batch[lig_mask], 0)
    timing['ligand_coeffs'] = timer.tick()
    #######################################
    prot_out = model.protein_model(batch, key="receptor", radius=False, all_atoms=model.args.all_atoms)
    timing['protein_model'] = timer.tick()
    #############################
    prot_anlm = []
    for i in range(offsets.shape[0]):
        prot_pos = batch['receptor'].pos - pocket_center - offsets[i]
        prot_mask = prot_pos.norm(dim=-1) < model.args.so3_radius
        prot_anlm.append(model.cache.local_to_global(prot_pos[prot_mask].unsqueeze(-2), prot_out[prot_mask]).sum(0))

    prot_anlm = torch.stack(prot_anlm)

    timing['protein_coeffs'] = timer.tick()
    #############################
    scores = lig_anlm.new_zeros(lig_anlm.shape[0], wigner_D.shape[0], prot_anlm.shape[0])
    for c, lig_anlm_ in enumerate(lig_anlm.unbind(0)):
        scores[c] = torch.einsum('ajm,takl,rlm,jk->rt', lig_anlm_, prot_anlm, wigner_D, model.cache.global_rbf_I / 100)
    timing['scoring'] = timer.tick()
    #############################
    scores_array = []
    csv = pd.read_csv(f"{args.pdbbind_dir}/{name}/{name}_poses.csv")
    for _, item in csv.iterrows():
        scores_array.append(scores[int(item.conf_idx), int(item.rot_idx), int(item.tr_idx)].item())
    np.save(f"{args.outdir}/{name}.npy", np.array(scores_array))
    #######################
    timing['all'] = 1000*(time.time() - master_start)
    return timing


def do_batch_T(batch, model):
    name = batch.name
    device = batch['ligand'].pos.device
    cache = model.tr_cache
    if not cache.initialized:
        cache.initialize(
            R_grid_spacing=model.args.fft_resolution, 
            R_grid_diameter=model.args.box_diameter,
        device=device)
    
    npz = dict(np.load(f"{args.pdbbind_dir}/{name}/{name}_poses.npz"))
    pocket_center = batch.full_rdkit_mol.GetConformer(0).GetPositions().mean(0)
    pocket_center = torch.from_numpy(pocket_center).to(device).float()
    withH_pos = batch['ligand'].pos.cpu()
    noH_pos = torch.from_numpy(npz['poses'][0])
    distmat = (withH_pos.unsqueeze(-2) - noH_pos.unsqueeze(-3)).norm(dim=-1)
    noH_idx = distmat.min(1)[1]
    num_atoms = batch['ligand'].pos.shape[0]
    num_confs = npz['confs'].shape[0]
    ###################3
    timing = {'name': name}
    master_start = time.time()
    timer = CudaTimer()
    ################
    big_batch = Batch.from_data_list([batch] * num_confs)    
    big_batch['ligand'].pos = lig_pos = torch.from_numpy(npz['confs'][:,noH_idx].reshape(-1, 3)).float().cuda()
   
    offsets = torch.from_numpy(npz['offsets']).float().to('cuda')
    rots = torch.from_numpy(npz['rots']).float()
    wigner_D = model.ligand_model.sh_irreps.D_from_matrix(rots).to(device).float()
    rots = rots.float().to(device)
    timing['preprocess'] = timer.tick()
    ######################
    ligand_out = model.ligand_model(big_batch, key="ligand", radius=True)
    ligand_out = ligand_out.reshape(num_confs, num_atoms, *ligand_out.shape[-3:])
    timing['ligand_model'] = timer.tick()
    ###############
    prot_pos = batch['receptor'].pos - pocket_center
    prot_out = model.protein_model(batch, key="receptor", radius=False, all_atoms=model.args.all_atoms)
    timing['protein_model'] = timer.tick()
    ################
    prot_mask = (prot_pos.abs() < model.args.box_diameter // 2).sum(1) == 3
    prot_fft = cache.render_fft(prot_pos[prot_mask], prot_out[prot_mask])  
    timing['protein_fft'] = timer.tick()
    ################
    lig_pos = lig_pos.reshape(num_confs, num_atoms, 3)
    lig_pos = torch.einsum('rxy,cpy->crpx', rots, lig_pos)
    lig_anlm = torch.einsum('rxy,cpany->crpanx', wigner_D, ligand_out)
    timing['ligand_coeffs'] = timer.tick()
    ###############
    ligand_fft = prot_fft.new_zeros(num_confs, rots.shape[0], model.args.num_channels, 
        cache.NK_grid, cache.NK_grid, cache.NK_grid) # craxyz
    for i in range(num_confs):
        for j in range(rots.shape[0]):
            ligand_fft[i,j] = cache.render_fft(lig_pos[i,j], lig_anlm[i,j])
    timing['ligand_fft'] = timer.tick()
    #################
    offset_translate = torch.exp(-1j * torch.einsum("xyza,pa->pxyz", cache.kxyz, offsets))    
    factor = cache.K_grid_spacing ** 3 * cache.NK_grid**1.5 / (2*np.pi)**(3/2)
    scores = torch.einsum('axyz,craxyz,txyz->crt', torch.conj(prot_fft), ligand_fft, offset_translate).real * factor**2 / 100
    timing['scoring'] = timer.tick()
    ###################
    scores_array = []
    csv = pd.read_csv(f"{args.pdbbind_dir}/{name}/{name}_poses.csv")
    for _, item in csv.iterrows():
        scores_array.append(scores[int(item.conf_idx), int(item.rot_idx), int(item.tr_idx)].item())
    np.save(f"{args.outdir}/{name}.npy", np.array(scores_array))
    timing['all'] = 1000*(time.time() - master_start)
    return timing
    
if __name__ == "__main__":
    main()

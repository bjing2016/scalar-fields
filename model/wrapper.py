import torch, wandb, os
import numpy as np
import pandas as pd
from collections import defaultdict
import pytorch_lightning as pl
from torch_geometric.data import HeteroData
from torch_scatter import scatter_mean, scatter_sum
from datasets.process_mols import (
    lig_feature_dims,
    rec_residue_feature_dims,
    rec_atom_feature_dims,
)
from .model import TensorProductConvModel
from scipy.spatial.transform import Rotation
from utils import so3fft
from utils.fft import cross_correlate, FFTCache
from utils.logging import get_logger
logger = get_logger(__name__)

def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log


def get_log_mean(log):
    out = {}
    for key in log:
        try:
            out[key] = np.mean(log[key])
        except:
            pass
    return out


class ModelWrapper(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self._log = defaultdict(list)

        self.protein_model = TensorProductConvModel(
            args,
            rec_residue_feature_dims[0],
            edge_features=0,
            atom_features=rec_atom_feature_dims[0] if args.all_atoms else None,
        )
        self.ligand_model = TensorProductConvModel(
            args,
            lig_feature_dims[0],
            edge_features=4,
        )

        self.cache = so3fft.SO3FFTCache(
            local_num_rbf=args.num_rbf,
            local_rbf_max=args.rbf_max,
            local_rbf_min=0,
            local_lmax=args.order,
            global_rbf_max=args.so3_radius,
            global_rbf_min=0,
            global_num_rbf=args.global_num_rbf,
            global_lmax=args.global_order,
            grid_diameter=2 * args.so3_radius,
            NR_grid=100,
            basis_type=args.basis_type,
            basis_cutoff=args.basis_cutoff,
        )

        self.tr_cache = FFTCache(
            R_basis_max=args.R_basis_max,
            R_basis_spacing=args.R_basis_spacing,
            buffer_factor=args.buffer_factor,
            rbf_max=args.rbf_max,
            num_rbf=args.num_rbf,
            sh_lmax=args.order,
            basis_type=args.basis_type,
            basis_cutoff=args.basis_cutoff,
        )
        self.iter_idx = 0

    def try_print_log(self):
        if (self.trainer.global_step + 1) % self.args.print_freq == 0:
            log = self._log
            log = {key: log[key] for key in log if "iter_" in key}

            log = gather_log(log, self.trainer.world_size)
            if self.trainer.is_global_zero:
                logger.info(str(get_log_mean(log)))
                if self.args.wandb:
                    wandb.log(get_log_mean(log))
            for key in list(log.keys()):
                if "iter_" in key:
                    del self._log[key]

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        log = self._log
        log["iter_" + key].extend(data)
        log[self.stage + "_" + key].extend(data)

    def on_train_epoch_end(self):
        log = self._log
        log = {key: log[key] for key in log if "train_" in key}
        log = gather_log(log, self.trainer.world_size)

        if self.trainer.is_global_zero:
            logger.info(str(get_log_mean(log)))
            if self.args.wandb:
                wandb.log(get_log_mean(log))

            path = os.path.join(os.environ["MODEL_DIR"], f"train_{self.trainer.current_epoch}.csv")
            pd.DataFrame(log).to_csv(path)
        for key in list(log.keys()):
            if "train_" in key:
                del self._log[key]

    def on_validation_epoch_end(self):
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        log = gather_log(log, self.trainer.world_size)
        if self.trainer.is_global_zero:
            logger.info(str(get_log_mean(log)))
            if self.args.wandb:
                wandb.log(get_log_mean(log))

            path = os.path.join(os.environ["MODEL_DIR"], f"val_{self.trainer.current_epoch}.csv")
            pd.DataFrame(log).to_csv(path)

        for key in list(log.keys()):
            if "val_" in key:
                del self._log[key]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        self.stage = "train"
        batch = HeteroData(batch)
        if self.args.mode == "mixed":
            if self.iter_idx % 2 == 0:
                loss = self.translation_iter(batch)
            else:
                loss = self.so3_iter(batch)
        elif self.args.mode == "T":
            loss = self.translation_iter(batch)
        elif self.args.mode == "R":
            loss = self.so3_iter(batch)
        self.try_print_log()
        self.iter_idx += 1
        return loss

    def validation_step(self, batch, batch_idx):
        self.stage = "val"
        batch = HeteroData(batch)
        # self.check_equivariance(batch)
        if self.args.mode == "mixed":
            if self.iter_idx % 2 == 0:
                loss = self.translation_iter(batch)
            else:
                loss = self.so3_iter(batch)
        elif self.args.mode == "T":
            loss = self.translation_iter(batch)
        elif self.args.mode == "R":
            loss = self.so3_iter(batch)

        self.iter_idx += 1
        self.try_print_log()

    def translation_iter(self, batch):
        device = batch["ligand"].pos.device
        cache = self.tr_cache

        if not cache.initialized:
            cache.initialize(
                R_grid_spacing=self.args.fft_resolution,
                R_grid_diameter=self.args.box_diameter,
                device=device,
            )

        bid = batch["ligand"].batch
        pid = batch["receptor"].batch
        batch_size = bid.max().item() + 1

        pocket_center = scatter_mean(batch["ligand"].pos, bid, 0)

        if self.stage == 'train' and self.args.rot_noise:
            random_rot = np.random.randn(3) * self.args.rot_noise
            random_rot = Rotation.from_rotvec(random_rot).as_matrix()
            random_rot = torch.from_numpy(random_rot).to(batch['ligand'].pos)
            batch['ligand'].pos = (batch['ligand'].pos - pocket_center[bid]) @ random_rot.T + pocket_center[bid]
        
        prot_pos = batch["receptor"].pos - pocket_center[pid]
        prot_mask = (prot_pos.abs() < self.args.box_diameter // 2).sum(-1) == 3
        prot_out = self.protein_model(
            batch, key="receptor", radius=False, all_atoms=self.args.all_atoms
        )

        if batch_size > 1:
            prot_fft = cache.render_fft(prot_pos[prot_mask], prot_out[prot_mask], sum=False)
            prot_fft = scatter_sum(prot_fft, pid[prot_mask], 0)
        else:
            prot_fft = cache.render_fft(prot_pos[prot_mask], prot_out[prot_mask]).unsqueeze(0)

        lig_pos = batch["ligand"].pos - pocket_center[bid]
        ligand_out = self.ligand_model(batch, key="ligand", radius=True)
        lig_mask = (lig_pos.abs() < self.args.box_diameter // 2).sum(-1) == 3
        if batch_size > 1:
            ligand_fft = cache.render_fft(lig_pos[lig_mask], ligand_out[lig_mask], sum=False)
            ligand_fft = scatter_sum(ligand_fft, bid[lig_mask], 0)
        else:
            ligand_fft = cache.render_fft(lig_pos[lig_mask], ligand_out[lig_mask]).unsqueeze(0)
        
        scaling = self.args.fft_resolution / self.args.cc_resolution
        field = cross_correlate(cache, protein=prot_fft, ligand=ligand_fft, scaling=scaling).sum(1) / 100

        size = field.shape[-1]
        idx = (size // 2) * (size**2 + size + 1)
        loss = torch.nn.functional.cross_entropy(
            field.view(batch_size, -1),
            torch.ones(batch_size, dtype=torch.long, device=device) * idx,
            reduction="none",
        )

        with torch.no_grad():
            if self.stage == "train":
                self.log("loss", loss)
                self.log("field_max", field.max(-1)[0].max(-1)[0].max(-1)[0])
                self.log("identity_entry", field.view(batch_size, -1)[:, idx])
                self.log("mean_entry", field.mean((-1, -2, -3)))
            self.log("name", batch.name)
            self.log("mode", ["trans"] * batch_size)
            self.log("prot_size", scatter_sum(0 * pid + 1, pid))
            self.log("lig_size", scatter_sum(0 * bid + 1, bid))
            self.log("prot_mask", scatter_sum(prot_mask.int(), pid))
            self.log("lig_mask", scatter_sum(lig_mask.int(), bid))
            self.log("prot_frac", scatter_mean(prot_mask.float(), pid))
            self.log("lig_frac", scatter_mean(lig_mask.float(), bid))

            argmax_idx = field.reshape(batch_size, -1).argmax(-1)
            rmsd = cache.rr.flatten()[argmax_idx]
            self.log("rmsd", rmsd)
            self.log("rmsd<2", rmsd < 2)
            self.log("rmsd<5", rmsd < 5)

        return loss.mean()

    def check_equivariance(self, batch):
        device = batch["ligand"].pos.device
        cache = self.cache
        bid = batch["ligand"].batch
        pid = batch["receptor"].batch

        lig_mean = scatter_mean(batch["ligand"].pos, bid, -2)
        lig_pos_orig = batch["ligand"].pos - lig_mean[bid]

        # if self.stage == 'train' or True:
        #     rot = so3fft.wigner_D(1, 0, - self.args.rot_pi * np.pi, 0, real=True, order='YZX').to(device)
        # else:
        #     angles = Rotation.random().as_euler('ZYZ')
        #     rot = so3fft.wigner_D(1, *angles, real=True, order='YZX').to(device)

        # batch['ligand'].pos = (lig_pos_orig @ rot.T) + lig_mean[bid]

        # prot_pos = batch['receptor'].pos - lig_mean[pid]
        lig_pos = batch["ligand"].pos - lig_mean[bid]
        ligand_out = self.ligand_model(batch, key="ligand", radius=True)
        # prot_out = self.protein_model(batch, key="receptor", radius=False)

        # prot_mask = prot_pos.norm(dim=-1) < self.args.so3_radius
        lig_mask = lig_pos.norm(dim=-1) < self.args.so3_radius

        # prot_anlm = cache.local_to_global(prot_pos[prot_mask].unsqueeze(-2), prot_out[prot_mask])
        # prot_anlm = scatter_sum(prot_anlm, pid[prot_mask], 0)

        lig_anlm = cache.local_to_global(lig_pos[lig_mask].unsqueeze(-2), ligand_out[lig_mask])
        lig_anlm = scatter_sum(lig_anlm, bid[lig_mask], 0)

        angles = Rotation.random().as_euler("ZYZ")
        rot = so3fft.wigner_D(1, *angles, real=True, order="YZX").to(device)
        batch["ligand"].pos = (lig_pos_orig @ rot.T) + lig_mean[bid]
        rot_ligand_out = self.ligand_model(batch, key="ligand", radius=True)
        rot_lig_pos = batch["ligand"].pos - lig_mean[bid]
        rot_lig_anlm = cache.local_to_global(
            rot_lig_pos[lig_mask].unsqueeze(-2), rot_ligand_out[lig_mask]
        )
        rot_lig_anlm = scatter_sum(rot_lig_anlm, bid[lig_mask], 0)

        ligand_out_rot = (
            ligand_out
            @ so3fft.irrep_wigner_D(cache.local_irreps, *angles, real=True, order="YZX")
            .to(device)
            .T
        )
        lig_pos_rot = lig_pos @ rot.T
        lig_anlm_rot = (
            lig_anlm
            @ so3fft.irrep_wigner_D(cache.global_irreps, *angles, real=True, order="YZX")
            .to(device)
            .T
        )

    def so3_iter(self, batch):
        device = batch["ligand"].pos.device

        cache = self.cache
        if cache.cached_C is None:
            cache.to(device)
            cache.precompute_offsets()

        bid = batch["ligand"].batch
        pid = batch["receptor"].batch
        batch_size = bid.max().item() + 1

        if self.stage == 'train' and self.args.tr_noise:
            batch["ligand"].pos += torch.randn(3, device=device) * self.args.tr_noise
        
        lig_mean = scatter_mean(batch["ligand"].pos, bid, -2)
        lig_pos_orig = batch["ligand"].pos - lig_mean[bid]

        if self.stage == "train" or self.args.no_val_randomize:
            rot = so3fft.wigner_D(1, 0, -self.args.rot_pi * np.pi, 0, real=True, order="YZX").to(device)
        else:
            angles = Rotation.random().as_euler("ZYZ")
            rot = so3fft.wigner_D(1, *angles, real=True, order="YZX").to(device)

        batch["ligand"].pos = (lig_pos_orig @ rot.T) + lig_mean[bid]

        prot_pos = batch["receptor"].pos - lig_mean[pid]
        lig_pos = batch["ligand"].pos - lig_mean[bid]

        ligand_out = self.ligand_model(batch, key="ligand", radius=True)
        prot_out = self.protein_model(
            batch, key="receptor", radius=False, all_atoms=self.args.all_atoms
        )

        prot_mask = prot_pos.norm(dim=-1) < self.args.so3_radius
        lig_mask = lig_pos.norm(dim=-1) < self.args.so3_radius

        prot_anlm = cache.local_to_global(prot_pos[prot_mask].unsqueeze(-2), prot_out[prot_mask])
        prot_anlm = scatter_sum(prot_anlm, pid[prot_mask], 0)

        lig_anlm = cache.local_to_global(lig_pos[lig_mask].unsqueeze(-2), ligand_out[lig_mask])
        lig_anlm = scatter_sum(lig_anlm, bid[lig_mask], 0)
        field = so3fft.so3_fft(
            prot_anlm, lig_anlm, cache.global_irreps, cache.global_rbf_I / 100, lmax=self.args.so3fft_lmax
        ).sum(
            -4
        )  # sum over channels

        if self.stage == "train":
            thetas = so3fft.get_thetas(field)
            logit_field = field + torch.log(torch.sin(thetas).abs()).view(-1, 1)
            idx = so3fft.index_field(logit_field, 0, self.args.rot_pi * np.pi, 0)
            N = field.shape[-1]
            idx = idx[-3] * N**2 + idx[-2] * N + idx[-1]

            loss = torch.nn.functional.cross_entropy(
                logit_field.view(batch_size, -1),
                torch.ones(batch_size, dtype=torch.long, device=device) * idx,
                reduction="none",
            )

        with torch.no_grad():
            if self.stage == "train":
                self.log("loss", loss)
                self.log("field_max", field.max(-1)[0].max(-1)[0].max(-1)[0])
                a, b, c = so3fft.index_field(logit_field, 0, self.args.rot_pi * np.pi, 0)
                self.log("identity_entry", field[..., a, b, c])
                self.log("mean_entry", field.mean((-1, -2, -3)))
            self.log("name", batch.name)
            self.log("mode", ["so3"] * batch_size)
            self.log("prot_size", scatter_sum(0 * pid + 1, pid))
            self.log("lig_size", scatter_sum(0 * bid + 1, bid))
            self.log("prot_mask", scatter_sum(prot_mask.int(), pid))
            self.log("lig_mask", scatter_sum(lig_mask.int(), bid))
            self.log("prot_frac", scatter_mean(prot_mask.float(), pid))
            self.log("lig_frac", scatter_mean(lig_mask.float(), bid))

            for i, field_ in enumerate(field.unbind(0)):
                a, b, c = so3fft.unindex_field(field_, field_.argmax().item())
                unrot = so3fft.wigner_D(1, a, b, c, real=True, order="YZX").to(device)
                rmsd = torch.square(lig_pos[bid == i] @ unrot.T - lig_pos_orig[bid == i])
                rmsd = rmsd.sum(-1).mean().item() ** 0.5
                self.log("rmsd", [rmsd])
                self.log("rmsd<2", [rmsd < 2])
                self.log("rmsd<5", [rmsd < 5])

        if self.stage == "train":
            return loss.mean()

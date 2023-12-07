import torch, e3nn
from scipy.special import spherical_jn
import e3nn.o3 as o3
import numpy as np


def RenderFFTFactory(ls_tensor, kxyz, grid):
    class RenderFFT(torch.autograd.Function):
        @staticmethod
        def forward(ctx, pos, anlm, sum):
            anlm = anlm * ls_tensor
            translate = torch.exp(-1j * torch.einsum("xyza,pa->pxyz", kxyz, pos)) 
            if sum:
                out = torch.einsum('panl,nlxyz,pxyz->axyz', anlm, grid, translate)
            else:
                out = torch.einsum('panl,nlxyz,pxyz->paxyz', anlm, grid, translate)
            ctx.save_for_backward(translate)
            ctx.sum = sum
            return out

        @staticmethod
        def backward(ctx, grad_output):
            (translate,) = ctx.saved_tensors
            if ctx.sum:
                d_anlm = torch.einsum("axyz,nlxyz,pxyz->panl", torch.conj(grad_output), grid, translate)
            else:
                d_anlm = torch.einsum("paxyz,nlxyz,pxyz->panl", torch.conj(grad_output), grid, translate)
            # d_anlm = torch.einsum('...xyz,nlxyz,pxyz->...pnl', torch.conj(grad_output), grid, translate)
            d_anlm = (d_anlm * ls_tensor).real
            return None, d_anlm, None

    return RenderFFT


class FFTCache:
    def __init__(
        self,
        R_basis_max=10,  # precomputed RBFs up to this distance
        R_basis_spacing=0.005,
        buffer_factor=10,  # Frankel transform capped for anti-ringing
        rbf_max=5,
        num_rbf=5,
        sh_lmax=4,
        basis_type='gaussian',
        basis_cutoff=False,
    ):
        self.R_basis_max = R_basis_max
        self.R_basis_spacing = R_basis_spacing
        self.buffer_factor = buffer_factor

        self.K_basis_max = 2 * np.pi / R_basis_spacing / buffer_factor
        self.K_basis_spacing = 2 * np.pi / R_basis_max / buffer_factor

        self.NR_basis = int(np.around(R_basis_max / R_basis_spacing))
        self.NK_basis = int(np.around(self.K_basis_max / self.K_basis_spacing))

        self.Rs_basis = np.linspace(0, self.R_basis_max, self.NR_basis + 1)[
            : self.NR_basis
        ]
        self.Ks_basis = np.linspace(0, self.K_basis_max, self.NK_basis + 1)[
            : self.NK_basis
        ]

        self.RBFs = (
            e3nn.math.soft_one_hot_linspace(
                torch.from_numpy(self.Rs_basis),
                0,
                rbf_max,
                number=num_rbf,
                basis=basis_type,
                cutoff=basis_cutoff,
            ).numpy().T
        )

        self.irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)

        self.rbf_transformed = {}

        for l in range(self.irreps.num_irreps):
            self.rbf_transformed[l] = (
                np.stack([
                    (spherical_jn(l, k * self.Rs_basis) * self.RBFs * self.Rs_basis**2).sum(-1) \
                    for k in self.Ks_basis
                ], -1) * self.R_basis_spacing * np.sqrt(2 / np.pi)
            )
            back = (
                np.stack(
                    [
                        (
                            spherical_jn(l, self.Ks_basis * r)
                            * self.rbf_transformed[l]
                            * self.Ks_basis**2
                        ).sum(-1)
                        for r in self.Rs_basis
                    ],
                    -1,
                )
                * self.K_basis_spacing
                * np.sqrt(2 / np.pi)
            )

            cosine_sims = (
                (self.RBFs * back).sum(-1)
                / np.sqrt(np.square(self.RBFs).sum(-1))
                / np.sqrt(np.square(back).sum(-1))
            )

            print("Checking RBF Frankel transform, l=", l, cosine_sims)

        self.ls = sum([[l] * (2 * l + 1) for l in range(self.irreps.num_irreps)], [])

        # self.K_grid_max = 2 * np.pi / R_grid_spacing
        self.num_rbf = num_rbf
        self.initialized = False

    def initialize(self, R_grid_spacing, R_grid_diameter, device="cpu"):
        

        ##### THE WAY IT USED TO BE######
        # self.R_grid_diameter = R_grid_diameter
        # self.K_grid_spacing = 2 * np.pi / R_grid_diameter
        # assert self.K_grid_max < self.K_basis_max
        # self.NR_grid = int(np.around(R_grid_diameter / self.R_grid_spacing))
        # self.NK_grid = int(np.around(self.K_grid_max / self.K_grid_spacing))
        # self.Rs_grid = np.linspace(
        #     -self.R_grid_diameter / 2, self.R_grid_diameter / 2, self.NR_grid + 1
        # )
        # self.Ks_grid = np.linspace(
        #     -self.K_grid_max / 2, self.K_grid_max / 2, self.NK_grid + 1
        # )
        ###################################
        self.R_grid_spacing = R_grid_spacing
        self.R_grid_diameter = R_grid_diameter
        self.NK_grid = self.NR_grid = int(np.around(R_grid_diameter / self.R_grid_spacing)) # + 1
        self.Rs_grid = np.linspace(
            -self.R_grid_diameter / 2, self.R_grid_diameter / 2, self.NR_grid + 1
        )[:self.NR_grid]
        self.Ks_grid = np.fft.fftshift(np.fft.fftfreq(self.NK_grid)) / R_grid_spacing * 2 * np.pi
        
        
        self.K_grid_spacing = self.Ks_grid[1] - self.Ks_grid[0]
        assert 2 * self.Ks_grid.max() < self.K_basis_max


        rx, ry, rz = np.meshgrid(
            self.Rs_grid, self.Rs_grid, self.Rs_grid, indexing="ij"
        )
        rxyz = np.stack([rx, ry, rz], -1)
        rr = np.sqrt(rx**2 + ry**2 + rz**2)

        kx, ky, kz = np.meshgrid(
            self.Ks_grid, self.Ks_grid, self.Ks_grid, indexing="ij"
        )
        kxyz = np.stack([kx, ky, kz], -1)
        ksh = o3.spherical_harmonics(
            self.irreps, torch.from_numpy(kxyz), normalize=True
        ).numpy()
        kr = np.sqrt(kx**2 + ky**2 + kz**2)

        interped = []

        for i, l in enumerate(self.ls):
            interped_ = np.stack(
                [
                    np.interp(
                        kr, self.Ks_basis, self.rbf_transformed[l][j]
                    )  # scipy.interpolate.interpn([self.Ks_basis],
                    #                          self.rbf_transformed[l][j], kr[...,None]) \
                    * ksh[..., i]
                    for j in range(self.num_rbf)
                ]
            )
            interped.append(interped_)
        
        self.grid = torch.from_numpy(np.stack(interped, 1)).to(
            device, dtype=torch.complex64
        )
        self.kxyz = torch.from_numpy(kxyz).to(device).float()
        self.rxyz = torch.from_numpy(rxyz).to(device).float()
        self.kr = torch.from_numpy(kr).to(device).float()
        self.rr = torch.from_numpy(rr).to(device).float()
        self.ls_tensor = (-1j) ** torch.tensor(self.ls, device=device)
        self.RenderFFT = RenderFFTFactory(self.ls_tensor, self.kxyz, self.grid)
        self.initialized = True
    """
    def to(self, device):
        self.grid = self.grid.to(device)
        self.kxyz = self.kxyz.to(device)
        self.rxyz = self.rxyz.to(device)
        self.kr = self.kr.to(device)
        self.rr = self.rr.to(device)
        self.ls_tensor = self.ls_tensor.to(device)
    """

    def render_fft(self, pos, anlm, sum=True):
        return self.RenderFFT.apply(pos, anlm, sum)
        """
        anlm = anlm * self.ls_tensor
        translate = torch.exp(-1j * torch.einsum('xyza,pa->pxyz', self.kxyz, pos))
        out = torch.einsum('...pnl,nlxyz,pxyz->...xyz', anlm, self.grid.to(anlm), translate)     # out[c,x,y,z] = sum_{n,l,p} a[c,p,n,l] * b[n,l,x,y,z] * c[p,x,y,z]   
        return out
        """
        # out = torch.einsum('...pnl,nlxyz->...pxyz', anlm , self.grid.to(anlm)) # out[c,p,x,y,z] = sum_{n, l} a[c,p,n,l] * b[n,l,x,y,z]     c * p * n * l * x * y * z
        # return torch.einsum('...pxyz,pxyz->...xyz', out, translate)   # out[x, y, z] = sum_p a[p,x,y,z] * b[p,x,y,z] p * x * y * z

    def render_field(self, pos, anlm):
        sh = o3.spherical_harmonics(
            self.irreps, self.rxyz[..., None, :] - pos, normalize=True
        ).float()
        r = torch.sqrt(torch.square(self.rxyz[..., None, :] - pos).sum(-1))
        fnr = np.stack(
            [np.interp(r.cpu().numpy(), self.Rs_basis, rbf) for rbf in self.RBFs], -1
        )
        fnr = torch.from_numpy(fnr).to(r)
        field = torch.einsum("xyzpl,panl->xyzpan", sh, anlm)
        field = torch.einsum("xyzpan,xyzpn->axyz", field, fnr)
        return field


def cross_correlate(cache, protein, ligand, scaling=1.0, sum=None):
    field = protein * torch.conj(ligand) * (2 * np.pi) ** (3 / 2)
    if sum: field = field.sum(sum)
    return ifft(cache, field, scaling=scaling)


def ifft(cache, field, scaling=1.0, complex=False):
    pad_k = int((scaling - 1.0) * cache.NK_grid // 2)
    if pad_k > 0:
        field = torch.nn.functional.pad(field, [pad_k] * 6)

    DIMS = (-1, -2, -3)
    field = (
        torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(field, dim=DIMS), dim=DIMS), dim=DIMS
        )
        * cache.K_grid_spacing**3
        * cache.NK_grid**3
        / (2 * np.pi) ** (3 / 2)
    )
    if complex:
        return field * scaling ** 3
    return field.real * scaling**3


def fft(cache, field):
    DIMS = (-1, -2, -3)
    field = (
        torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(field, dim=DIMS), dim=DIMS), dim=DIMS
        )
        * cache.R_grid_spacing**3
        / (2 * np.pi) ** (3 / 2)
    )
    return field


def get_random_field(cache, N_channels, N_points):
    pos = torch.randn(N_points, 3, device=cache.grid.device)
    anlm = torch.randn(
        N_channels,
        N_points,
        cache.RBFs.shape[0],
        cache.irreps.dim,
        device=cache.grid.device,
    )
    return pos, anlm


def cosine_sim(field1, field2):
    return (field2 * field1).sum() / torch.norm(field1) / torch.norm(field2)


if __name__ == "__main__":
    cache = FFTCache(num_rbf=3, rbf_max=1, sh_lmax=1)
    cache.initialize(R_grid_diameter=10, device="cuda")

    prot_pos, prot_anlm = get_random_field(cache, 1, 10)
    lig_pos, lig_anlm = get_random_field(cache, 1, 10)
    """
    prot_anlm = prot_anlm.to(torch.complex64)
    prot_anlm_fast = prot_anlm * 1.0
    
    prot_anlm.requires_grad_() 
    prot_field = cache.render_fft(prot_pos, prot_anlm, faster=False)
    prot_field.retain_grad() 
    cc_field = cross_correlate(cache, protein=prot_field, ligand=prot_field)
    #cc_field[0,0,0,0].backward()
    prot_field[0,0,0,0].backward()
    
    prot_anlm_fast.requires_grad_() 
    prot_field_fast = cache.render_fft(prot_pos, prot_anlm_fast, faster=True)
    prot_field_fast.retain_grad() 
    cc_field_fast = cross_correlate(cache, protein=prot_field_fast, ligand=prot_field_fast)
    #cc_field_fast[0,0,0,0].backward()
    prot_field_fast[0,0,0,0].backward()
    
    print(prot_anlm.grad[0,0,0])
    print(prot_anlm_fast.grad[0,0,0])
    
    import pdb
    pdb.set_trace()
    
    
    print(prot_field.grad[0,0,0])
    print(prot_field_fast.grad[0,0,0])
    
    """

    prot_field = cache.render_fft(prot_pos, prot_anlm)
    lig_field = cache.render_fft(lig_pos, lig_anlm)

    cc_field = cross_correlate(cache, prot_field, lig_field, scaling=2)
    prot_ref_field = cache.render_field(prot_pos, prot_anlm)
    lig_ref_field = cache.render_field(lig_pos, lig_anlm)
    lig_field_shifted = cache.render_field(
        lig_pos + torch.ones(3).to(lig_pos) * cache.R_grid_spacing, lig_anlm
    )

    print(
        (prot_ref_field * lig_field_shifted).sum() * cache.R_grid_spacing**3,
        cc_field[0, 51 * 2, 51 * 2, 51 * 2],
    )

    print(
        (prot_ref_field * lig_ref_field).sum() * cache.R_grid_spacing**3,
        cc_field[0, 50 * 2, 50 * 2, 50 * 2],
    )
    exit()
    import matplotlib.pyplot as plt

    plt.imshow(prot_ref_field[0, 50].cpu().numpy())
    plt.savefig("prot.png")
    plt.clf()

    plt.imshow(lig_ref_field[0, 50].cpu().numpy())
    plt.savefig("lig.png")
    plt.clf()

    plt.imshow(cc_field[0, 50 * 2].cpu().numpy())
    plt.savefig("cc.png")
    plt.clf()

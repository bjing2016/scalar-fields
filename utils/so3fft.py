import e3nn.o3 as o3
spherical_harmonics = o3
import utils.spherical_harmonics as spherical_harmonics
import torch, e3nn, tqdm, functools
import numpy as np
import matplotlib.pyplot as plt

@functools.lru_cache(maxsize=None)
def su2_generators(j) -> torch.Tensor:
    m = torch.arange(-j, j)
    raising = torch.diag(-torch.sqrt(j * (j + 1) - m * (m + 1)), diagonal=-1)

    m = torch.arange(-j + 1, j + 1)
    lowering = torch.diag(torch.sqrt(j * (j + 1) - m * (m - 1)), diagonal=1)

    m = torch.arange(-j, j + 1)
    return torch.stack([
        0.5 * (raising + lowering),  # y
        torch.diag(1j * m),          # z
        -0.5j * (raising - lowering),# x
    ], dim=0)

def wigner_d(l, theta): 
    X = su2_generators(l)
    return torch.matrix_exp(theta * X[0]) 

def wigner_D(l, phi, theta, psi, real=False, order='YZX'):
    
    X = su2_generators(l)
    try:
        X = X.to(theta.device)
    except: # is a scalar
        pass

    if isinstance(phi, torch.Tensor):
        phi = phi.unsqueeze(-1).unsqueeze(-1)
    if isinstance(theta, torch.Tensor):
        theta = theta.unsqueeze(-1).unsqueeze(-1)
    if isinstance(psi, torch.Tensor):
        psi = psi.unsqueeze(-1).unsqueeze(-1)
    
    if real:
        Q = o3._wigner.change_basis_real_to_complex(l).to(X.device)
        X = torch.conj(Q.T) @ X @ Q
    if order == 'YZX':
        out = torch.matrix_exp(phi * X[1]) @ torch.matrix_exp(theta * X[0]) @ torch.matrix_exp(psi * X[1]) 
    elif order == 'XYZ':
        out = torch.matrix_exp(phi * X[2]) @ torch.matrix_exp(theta * X[1]) @ torch.matrix_exp(psi * X[2]) 
    if real:
        out = out.real
    return out

def irrep_wigner_D(irreps, phi, theta, psi, real=True, order='XYZ'):
    batch_dims = []
    device = 'cpu'
    if isinstance(theta, torch.Tensor):
        batch_dims = list(theta.shape)
        device = theta.device
    D = torch.zeros(batch_dims + [irreps.dim, irreps.dim], device=device)
    for l, _ in enumerate(irreps.slices()):
        D[...,l**2:(l+1)**2,l**2:(l+1)**2] = wigner_D(l, phi, theta, psi, real=real, order=order)
    return D

def pyramid_sum(arrs):
    L = int((arrs[-1].shape[-1] - 1)//2)
    out = 0
    
    for l in range(L+1):
        pad = torch.nn.functional.pad(arrs[l], [L-l]*6)
        out = out + pad
    return out

def so3_fft(anlm, bnlm, irreps, rbf_I, lmax=10, sum=None):

    Ts = []
        
    for l, slice in enumerate(irreps.slices()):
        anlm_sl = anlm[...,slice]
        bnlm_sl = bnlm[...,slice]
        Q_sl = o3._wigner.change_basis_real_to_complex(l, device=anlm.device)
        # print(Q_sl.dtype)
        anlm_sl = anlm_sl.to(torch.complex64) @ Q_sl.T.to(torch.complex64)
        bnlm_sl = bnlm_sl.to(torch.complex64) @ Q_sl.T.to(torch.complex64)
        
        Ilmn = torch.einsum('...am,...bn,ab->...mn', anlm_sl, torch.conj(bnlm_sl), rbf_I.to(torch.complex64))
        dlmn = wigner_d(l, np.pi/2).to(anlm.device)
        Tlmn = torch.einsum('mh,hn,...mn->...mhn', dlmn, dlmn, Ilmn)
        
        Ts.append(Tlmn)
    
    T = pyramid_sum(Ts).flip((-1, -2, -3))
    
    pad = lmax - irreps.num_irreps + 1
    T = torch.nn.functional.pad(T, [pad]*6)
    dims = (-1,-2,-3)
    field = torch.fft.ifftn(torch.fft.ifftshift(T, dim=dims), dim=dims) * (T.shape[-1]) ** 3
    return field.real

def get_rbf_gram_matrix(rbf_func, r_max):
    rbf_r = torch.linspace(0, r_max, 10000) # buffer of 2x should be safe
    rbf_I = rbf_func(rbf_r)
    rbf_I = (rbf_I.unsqueeze(-1) * rbf_I.unsqueeze(-2) * rbf_r[:,None,None]**2).sum(0) * (rbf_r[1] - rbf_r[0])
    return rbf_I

def cartesian_render(Rs_grid, irreps, rbf_func, anlm, origin=None):
    
    xyz = torch.stack(torch.meshgrid(Rs_grid, Rs_grid, Rs_grid, indexing='ij'), -1)
    if origin is not None:
        xyz = xyz - origin[...,None,None,None,:]
    sh = spherical_harmonics.spherical_harmonics(irreps, xyz, normalize=True)
    rbf = rbf_func(torch.norm(xyz, dim=-1))

    L = anlm.shape[-1]
    out = 0
    for l in range(L):
        out += torch.einsum('...xyz,...xyza,...a->...xyz', sh[...,l], rbf, anlm[...,l])
        
    return out, Rs_grid[1] - Rs_grid[0]

class SO3FFTCache:
    def __init__(
        self, 
        local_rbf_min = 0,
        local_rbf_max = 1,
        local_num_rbf = 3,
        local_lmax = 4,
        global_rbf_min = 0,
        global_rbf_max = 5,
        global_num_rbf = 20,
        global_lmax = 20,
        grid_diameter = 10,
        NR_grid = 100,
        basis_type = "gaussian",
        basis_cutoff = False,
    ):
        self.Rs_grid = torch.linspace(-grid_diameter / 2, grid_diameter / 2, NR_grid + 1)

        self.local_num_rbf = local_num_rbf
        self.local_irreps = o3.Irreps.spherical_harmonics(lmax=local_lmax)
        self.local_rbf_func = lambda x: e3nn.math.soft_one_hot_linspace(x, local_rbf_min, local_rbf_max, 
                        number=local_num_rbf, basis=basis_type, cutoff=basis_cutoff)
        self.local_rbf_I = get_rbf_gram_matrix(self.local_rbf_func, 2*local_rbf_max) # is a tensor

        self.global_num_rbf = global_num_rbf
        self.global_irreps = o3.Irreps.spherical_harmonics(lmax=global_lmax)
        self.global_rbf_func = lambda x: e3nn.math.soft_one_hot_linspace(x, global_rbf_min, global_rbf_max, 
                        number=global_num_rbf, basis=basis_type, cutoff=basis_cutoff)
        self.global_rbf_I = get_rbf_gram_matrix(self.global_rbf_func, 2*global_rbf_max)
        G_inv = torch.linalg.inv(self.global_rbf_I)
        self.G_inv = torch.block_diag(*[G_inv]*(global_lmax+1)**2)

        self.offsets = torch.linspace(0, self.Rs_grid[-1], NR_grid // 2 + 1)
        self.cached_C = None
        
    def precompute_offsets(self):
        xyz = torch.stack(torch.meshgrid(self.Rs_grid, self.Rs_grid, self.Rs_grid, indexing='ij'), -1)
        self.sh = spherical_harmonics.spherical_harmonics(self.global_irreps, xyz, normalize=True)
        self.rbf = self.global_rbf_func(torch.norm(xyz, dim=-1))
        
        NR_grid = self.Rs_grid.shape[0] - 1
        
        arrs = []
        for off in tqdm.tqdm(self.offsets):
            # print(off)
            arrs.append(self.get_offset(off))
        self.cached_C = torch.stack(arrs)
    
    def get_offset(self, z_offset):
        
        if self.cached_C is not None:
            idx = z_offset / self.offsets[1]
            idx = torch.clamp(torch.round(idx), 0, len(self.offsets)-1).long()
            return self.cached_C[idx]

        xyz = torch.stack(torch.meshgrid(self.Rs_grid, self.Rs_grid, self.Rs_grid, indexing='ij'), -1)
        sh = spherical_harmonics.spherical_harmonics(self.global_irreps, xyz, normalize=True)
        rbf = self.global_rbf_func(torch.norm(xyz, dim=-1))
        
        offset_xyz = torch.stack(torch.meshgrid(self.Rs_grid, self.Rs_grid, self.Rs_grid - z_offset, indexing='ij'), -1)
        offset_sh = spherical_harmonics.spherical_harmonics(self.local_irreps, offset_xyz, normalize=True)
        offset_rbf = self.local_rbf_func(torch.norm(offset_xyz, dim=-1))
        
        #C = torch.einsum('xyza,xyzb,xyzc,xyzd->abcd', sh, rbf, offset_rbf, offset_sh)

        C_shape = [sh.shape[-1], rbf.shape[-1], offset_rbf.shape[-1], offset_sh.shape[-1]]
        C = sh.new_zeros(C_shape)

        for a in range(C_shape[0]):
            C[a] = torch.einsum('xyz,xyzb,xyzc,xyzd->bcd', sh[...,a], rbf, offset_rbf, offset_sh)

            
        dx = self.Rs_grid[1] - self.Rs_grid[0]
        C = C.reshape(self.global_irreps.dim * self.global_num_rbf, -1) * dx ** 3
        return C

    def local_to_global(self, pos, anlm):
        x, y, z = pos.unbind(-1)
        z_offset = torch.linalg.norm(pos, dim=-1)
        theta = torch.arccos(z / z_offset)
        phi = torch.atan2(y, x)

        batch_dims = anlm.shape[:-2]
        anlm_rot = anlm @ irrep_wigner_D(self.local_irreps, 0, -theta, -phi).transpose(-1, -2)
        C = self.get_offset(z_offset)
        
        anlm_new = self.G_inv @ C @ anlm_rot.reshape(*batch_dims, self.local_irreps.dim * self.local_num_rbf, 1)
        anlm_new = anlm_new.reshape(*batch_dims, self.global_irreps.dim, self.global_num_rbf).transpose(-1, -2)
        anlm_new = anlm_new @ irrep_wigner_D(self.global_irreps, phi, theta, 0).transpose(-1, -2)

        return anlm_new

    def to(self, device):
        for key in self.__dict__:
            try:
                self.__dict__[key] = self.__dict__[key].to(device)
            except:
                pass

def get_thetas(field):
    N = field.shape[-1]
    eta = 2 * np.pi * torch.arange(N, device=field.device) / N
    return np.pi - eta


    ###
    # t = 0, pi/2, pi, 3pi/2
    # freq = 0, -1, +1, -2 = +2


    # t = 0, pi/3, 2pi/3
    # freq = 0, -1, +1
    ##sum e^{m\phi}...

def index_field(field, phi, theta, psi):
    N = field.shape[-1]
    xi_idx = eta_idx = om_idx = None, None, None
    if phi is not None:
        xi = phi - np.pi/2
        xi_idx = int(np.round((N * xi / (2*np.pi)) % N))
    if theta is not None:
        eta = np.pi - theta
        eta_idx = int(np.round((N * eta / (2*np.pi)) % N))
    if psi is not None:
        om = psi - np.pi/2
        om_idx = int(np.round((N * om / (2*np.pi)) % N))
    return xi_idx, eta_idx, om_idx

def unindex_field(field, idx=None, xi_idx=None, eta_idx=None, om_idx=None):
    N = field.shape[-1]
    if idx is not None:
        xi_idx = int(idx // N**2)
        eta_idx = int(idx // N) % N
        om_idx = int(idx % N)
    xi = 2 * np.pi * xi_idx / N
    eta = 2 * np.pi * eta_idx / N
    om = 2 * np.pi * om_idx / N
    
    phi = xi + np.pi/2
    theta = np.pi - eta
    psi = om + np.pi/2
    return phi, theta, psi

if __name__ == "__main__":
    cache = SO3FFTCache(
        local_rbf_min=0,
        local_rbf_max=5,
        local_num_rbf=3,
        local_lmax=2, 
        global_rbf_min=0,
        global_rbf_max = 25,
        global_num_rbf = 2,
        global_lmax = 2,
        grid_diameter = 50
    )
    device = 'cuda'
    cache.to(device)
    cache.precompute_offsets()

    num_channels = 1
    num_particles = 10
    torch.manual_seed(137)
    anlm = torch.randn(num_channels, num_particles, cache.local_num_rbf, cache.local_irreps.dim, device=device) 
    dummy_pos = torch.randn(num_particles, 3, device=device) * 5
    cartesian_field, dx = cartesian_render(cache.Rs_grid, cache.local_irreps, cache.local_rbf_func, anlm, origin=dummy_pos)
    cartesian_field = cartesian_field.sum(-4) # sum over points
    
    print((cartesian_field**2).sum() * dx ** 3)
    
    for i in range(num_channels):
        plt.imshow(cartesian_field[i,50].cpu()); plt.colorbar(); plt.savefig(f'local{i}.png'); plt.clf()
    
    anlm_global = cache.local_to_global(dummy_pos, anlm).sum(-3) # sum over points
    cartesian_field, dx = cartesian_render(cache.Rs_grid, cache.global_irreps, cache.global_rbf_func, anlm_global)
    print((cartesian_field**2).sum() * dx ** 3)
    
    for i in range(num_channels):
        plt.imshow(cartesian_field[i,50].cpu()); plt.colorbar(); plt.savefig(f'global{i}.png'); plt.clf()
    
    so3_field = so3_fft(anlm_global, anlm_global, cache.global_irreps, cache.global_rbf_I, lmax=10).sum(-4) # sum over channels
    print(torch.einsum('cam,cbm,ab', anlm_global, anlm_global, cache.global_rbf_I))

    so3_field_copy = so3_field.new_zeros(*so3_field.shape)

    for i in tqdm.trange(21):
        for j in range(21):
            for k in range(21):
                a, b, c = unindex_field(so3_field, None, i, j, k)
                D = irrep_wigner_D(cache.global_irreps, a, b, c, real=True, order='YZX').to(anlm_global.device)
                so3_field_copy[i,j,k] = torch.einsum('cam,cbm,ab', anlm_global, anlm_global @ D.T, cache.global_rbf_I)
                print(so3_field_copy[i,j,k], so3_field[i,j,k])

    import pdb
    pdb.set_trace()

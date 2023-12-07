from argparse import ArgumentParser
import subprocess, os


def parse_train_args():
    parser = ArgumentParser()

    ## General training args
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--overfit", action='store_true')
    parser.add_argument("--accumulate_grad", type=int, default=1)
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--max_protein_len", type=int, default=float('nan'))
    parser.add_argument("--max_lig_size", type=int, default=float('nan'))

    parser.add_argument("--ablate_chemistry", action='store_true')
    ## Model training args
    parser.add_argument("--rot_pi", type=float, default=0.25)
    parser.add_argument("--no_val_randomize", action='store_true')
    parser.add_argument("--rot_noise", type=float, default=None)
    parser.add_argument("--tr_noise", type=float, default=None)
    
    ## Mode args
    parser.add_argument("--mode", type=str, choices=['T', 'R', 'mixed'], default='mixed')
    parser.add_argument("--all_atoms", action='store_true')
    
    ## Logging args
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="default")


    ## Common FFT args
    parser.add_argument("--num_rbf", type=int, default=3)
    parser.add_argument("--rbf_max", type=float, default=5)
    parser.add_argument("--num_channels", type=int, default=5)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--basis_type", type=str, default="gaussian")
    parser.add_argument("--basis_cutoff", action="store_true")

    
    ## SO3FFT args
    parser.add_argument('--so3_radius', type=float, default=20.)
    parser.add_argument("--global_num_rbf", type=int, default=25)
    parser.add_argument("--global_order", type=int, default=10)
    parser.add_argument("--so3fft_lmax", type=int, default=50)
    
    ## Translational FFT args
    parser.add_argument("--R_basis_max", type=float, default=10)
    parser.add_argument("--R_basis_spacing", type=float, default=0.005)
    parser.add_argument("--buffer_factor", type=float, default=10)
    parser.add_argument("--box_diameter", type=float, default=40)
    parser.add_argument("--fft_resolution", type=float, default=1)
    parser.add_argument("--cc_resolution", type=float, default=1)
    
    ## Model args
    parser.add_argument("--conv_layers", type=int, default=6)
    parser.add_argument("--ns", type=int, default=16)
    parser.add_argument("--nv", type=int, default=4)
    parser.add_argument("--fc_dim", type=int, default=128)

    parser.add_argument(
        "--radius_emb_type",
        type=str,
        choices=["sinusoidal", "gaussian"],
        default="gaussian",
    )
    parser.add_argument("--radius_emb_dim", type=int, default=100)
    parser.add_argument("--radius_emb_max", type=float, default=50)

    ## PDBBind args
    parser.add_argument(
        "--pdbbind_dir", type=str, default="data/PDBBind_processed"
    )
    parser.add_argument(
        "--train_split_path", type=str, default="splits/timesplit_no_lig_overlap_train"
    )
    parser.add_argument(
        "--val_split_path", type=str, default="splits/timesplit_no_lig_overlap_val"
    )
    parser.add_argument("--receptor_radius", type=float, default=30)
    parser.add_argument("--c_alpha_max_neighbors", type=int, default=10)
    parser.add_argument('--atom_radius', type=float, default=5)
    parser.add_argument('--atom_max_neighbors', type=int, default=8)

    args = parser.parse_args()
    os.environ["MODEL_DIR"] = os.path.join("workdir", args.run_name)
    os.environ["WANDB_LOGGING"] = str(int(args.wandb))

    return args

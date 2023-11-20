import models.SpectralRxnDiffusion as SpectralRxnDiffusion
import models.FiniteDiffRxnDiffusion as FiniteDiffRxnDiffusion
from utils.diffusion_utils import plot_T, plot_rxn_diffusion, plot_calcium
import numpy as np


def get_ca_init_loc():
    # VDCC
    vdcc_loc = 2.35  # um

    return vdcc_loc


def get_ca_init_idx(n_space_pts):
    vdcc_loc = get_ca_init_loc()

    z = 4  # um
    dx = z / n_space_pts

    return int(vdcc_loc / dx)


def main():
    n_time_pts = 101  # number of time points
    n_space_pts = 150  # 150  # number of spatial points
    ca_init_idx = get_ca_init_idx(n_space_pts)

    # Finite Difference (calbindin reactions)
    fd = FiniteDiffRxnDiffusion.FiniteDiffRxnDiffusion(
        n_spatial_locs=n_space_pts,
        n_time_pts=n_time_pts,
        impulse_idx=ca_init_idx,
    )
    fd.simulate_diffusion()
    n_eigenmodes = 300

    # Spectral Method (calbindin reactions)
    sd = SpectralRxnDiffusion.SpectralRxnDiffusion(
        n_spatial_locs=n_space_pts,
        n_time_pts=n_time_pts,
        impulse_idx=ca_init_idx,
        n_eigenmodes=n_eigenmodes,
    )

    sd.solve_dTdt(
        save_dir=f"../data/spectral-diffusion/eigenmode-exps/{n_eigenmodes}/T.npy"
    )
    sd.simulate_diffusion(
        # save_dir=f"../data/spectral-diffusion/eigenmode-exps/{n_eigenmodes}/u.npy"
    )

    # FINITE DIFFERENCE DIFFUSION
    # fd.simulate_diffusion()
    # fd.plot_diffusion([0, 1, 5, 20, 40, 50, 100])
    t = [0, 1, 5, 20, 40, 50, 100]
    fd.plot_finndiff_vs_spect(sd)



if __name__ == "__main__":
    main()

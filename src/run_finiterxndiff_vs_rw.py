import models.SpectralRxnDiffusion as SpectralRxnDiffusion
import models.FiniteDiffRxnDiffusion as FiniteDiffRxnDiffusion
from utils.diffusion_utils import plot_T, plot_rxn_diffusion, plot_calcium
import numpy as np
import utils.PlotMultiruns as pm


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
    n_space_pts = 101  # 150  # number of spatial points
    ca_init_idx = get_ca_init_idx(n_space_pts)

    # Finite Difference (calbindin reactions)
    fd = FiniteDiffRxnDiffusion.FiniteDiffRxnDiffusion(
        n_spatial_locs=n_space_pts,
        n_time_pts=n_time_pts,
        impulse_idx=ca_init_idx,
    )
    fd.simulate_diffusion()

    plotter = pm.PlotMultiRuns(
        rw_dir="../data/eme-validation/random-walk/20231120_111038/rw-run-{}.csv",  # "/Users/margotwagner/diffusion-model/data/eme-validation/random-walk/20230118_192600/rw-run-{}.csv",
        eme_dir="../data/eme-validation/markov-eme/20231120_111028/eme-run-{}.csv",  # "/Users/margotwagner/diffusion-model/data/eme-validation/markov-eme/20230405_105433_2/eme-run-{}.csv",
        n_runs=10,
        n_spatial_locs=101,
        n_time_pts=100,  # TODO: make automatic
        particle_start_loc=58,
        n_particles=50,
        plot_eme=False,  # TODO: fix LHS entry
        plot_rw=True,
    )

    plotter.plot_rw_eme_vs_finndiff(fd)




if __name__ == "__main__":
    main()
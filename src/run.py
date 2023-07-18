import models.SpectralRxnDiffusion as SpectralRxnDiffusion
import models.FiniteDiffRxnDiffusion as FiniteDiffRxnDiffusion


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
    n_space_pts = 150  # number of spatial points
    ca_init_idx = get_ca_init_idx(n_space_pts)

    # Finite Difference (calbindin reactions)
    fd = FiniteDiffRxnDiffusion.FiniteDiffRxnDiffusion(
        n_spatial_locs=n_space_pts,
        n_time_pts=n_time_pts,
        impulse_idx=ca_init_idx,
    )

    # Spectral Method (calbindin reactions)
    sd = SpectralRxnDiffusion.SpectralRxnDiffusion(
        n_spatial_locs=n_space_pts,
        n_time_pts=n_time_pts,
        impulse_idx=ca_init_idx,
        n_eigenmodes=n_space_pts,
    )

    # FINITE DIFFERENCE DIFFUSION
    fd.simulate_diffusion()
    fd.plot_diffusion([0, 1, 5, 20, 40, 50, 100])

    #fd.simulate_rxn_diffusion()
    #fd.plot_rxn_diffusion([0, 1, 5, 20, 40, 50, 100])

    # sd_rxn.solve_dTdt()
    # sd_rxn.solve_u()
    # sd_rxn.plot_T()
    # sd_rxn.plot_u()

    # Spectral Diffusion No Reactions
    # sd.plot(sd_u, [0, 1, 5, 20, 40, 50, 99])
    # sd.plot([1, 5, 20, 40, 50, 99], ylim=[0, 0.5], xlim=[0, 4])

    # Finite Differencing with Reactions
    # fd_rxn.plot([ca, calb, ca_calb], [0, 1, 2, 3, 4, 5, 10, 20, 40, 50, 99])
    # fd_rxn.plot([ca, calb, ca_calb],[1, 5, 10, 20, 25, 29])
    # fd_rxn.plot([ca, calb, ca_calb], [1, 2, 3, 4])

    # Spectral Diffusion with Reactions
    # sd_rxn.plot([0, 1])


if __name__ == "__main__":
    main()

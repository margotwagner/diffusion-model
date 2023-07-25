import models.SpectralRxnDiffusion as SpectralRxnDiffusion


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

    # Spectral Method (calbindin reactions)
    sd = SpectralRxnDiffusion.SpectralRxnDiffusion(
        n_spatial_locs=n_space_pts,
        n_time_pts=n_time_pts,
        impulse_idx=ca_init_idx,
        n_eigenmodes=75,
    )

    # SPECTRAL DIFFUSION
    # sd.simulate_diffusion()
    # sd.plot_diffusion([0, 1, 5, 20, 40, 50, 100])

    sd.solve_dTdt(save_dir="../data/spectral-diffusion/08202023/T.npy")
    sd.simulate_rxn_diffusion(save_dir="../data/spectral-diffusion/08202023/u.npy")


if __name__ == "__main__":
    main()

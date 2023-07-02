import FiniteDiffNoRxns
import SpectralDiffNoRxns

def main():
    vdcc_loc = 2.35  # um

    n_spatial_locs = 100
    dx = 4 / n_spatial_locs

    vdcc_loc_idx = int(vdcc_loc / dx)

    # Finite Difference (no reactions)
    fd = FiniteDiffNoRxns.FiniteDiffNoRxns(
        n_particles=50,
        n_spatial_locs=n_spatial_locs,
        n_time_pts=1000,
        particle_start_loc=vdcc_loc_idx,
    )

    # Spectral Diffusion (no reactions)
    sd = SpectralDiffNoRxns.SpectralDiffNoRxns(
        n_particles=50,
        n_spatial_locs=n_spatial_locs,
        n_time_pts=1000,
        particle_start_loc=vdcc_loc_idx,
        n_eigenmodes=n_spatial_locs,
    )


    #fd_u = fd.simulate()
    sd_u = sd.simulate()

    #fd.plot(fd_u, [0, 1, 5, 20, 40, 50, 99])
    #fd.plot(fd_u, [5, 20, 40, 50, 99])

    #sd.plot(sd_u, [0, 1, 5, 20, 40, 50, 99])
    sd.plot(sd_u, [5, 20, 40, 50, 99])




if __name__ == "__main__":
    main()

import utils.RunMultiruns as rm

def get_ca_init_loc():
    # VDCC
    vdcc_loc = 2.35  # um

    return vdcc_loc


def get_ca_init_idx(n_space_pts):
    vdcc_loc = get_ca_init_loc()

    z = 4  # um
    dx = z / n_space_pts

    return int(vdcc_loc / dx)

def main(
    n_runs,
    n_particles,
    n_spatial_locs,
    n_time_pts,
    particle_start_loc,
    run_type,  # type of simulation to run
    scaling_factor=1.0,
):

    if run_type == "rw":
        multi_rw = rm.RunMultiruns(
            n_runs=n_runs,
            n_particles=n_particles,
            n_spatial_locs=n_spatial_locs,
            n_time_pts=n_time_pts,
            particle_start_loc=particle_start_loc,
            scaling_factor=scaling_factor,
        )

        multi_rw.run_multi_rw()

    elif run_type == "eme":
        multi_eme = rm.RunMultiruns(
            n_runs=n_runs,
            n_particles=n_particles,
            n_spatial_locs=n_spatial_locs,
            n_time_pts=n_time_pts,
            particle_start_loc=particle_start_loc,
            scaling_factor=scaling_factor,
        )

        multi_eme.run_multi_eme()

    else:
        print("Invalid run_type. Please choose 'rw' or 'eme'.")


if __name__ == "__main__":
    n_time_pts = 100  # number of time points
    n_space_pts = 101  # 150  # number of spatial points
    ca_init_idx = get_ca_init_idx(n_space_pts)

    main(
        n_runs=10,
        n_particles=50,
        n_spatial_locs=n_space_pts,
        n_time_pts=n_time_pts,
        particle_start_loc=ca_init_idx,
        run_type="eme",  # type of simulation to run
        scaling_factor=2,
    )
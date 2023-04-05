import RunMultiruns as rm
import numpy as np


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
        )

        multi_eme.run_multi_eme()

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
    main(
        n_runs=1000,
        n_particles=50,
        n_spatial_locs=11,
        n_time_pts=1000,
        particle_start_loc=5,
        run_type="eme",  # type of simulation to run
        scaling_factor=np.sqrt(2),
    )

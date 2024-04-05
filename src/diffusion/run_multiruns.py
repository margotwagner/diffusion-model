"""Run multiple diffusion simulations using either random walk or EME method.

Usage: python3 ./run_multiruns.py
"""

__author__ = ["Margot Wagner"]
__contact__ = "mwagner@ucsd.edu"
__date__ = "2023/06/13"

import sys

sys.path.append("../../src/")
import utils.RunMultiruns as rm


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
    main(
        n_runs=10,
        n_particles=50,
        n_spatial_locs=101,
        n_time_pts=100,
        particle_start_loc=58,
        run_type="eme",  # type of simulation to run
        scaling_factor=2,
    )
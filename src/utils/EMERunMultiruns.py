import models.EigenmarkovDiffusion as emd
import os
import numpy as np


class EMERunMultiruns:
    def __init__(
        self,
        n_runs,
        n_particles,
        n_spatial_locs,
        n_time_pts,
        impulse_idx,
    ):
        self.n_runs = n_runs
        self.n_particles = n_particles
        self.n_spatial_locs = n_spatial_locs
        self.n_time_pts = n_time_pts
        self.impulse_idx = impulse_idx
        self.scaling_factor = 0.5

    def run(self, normalize=False):

        eigenmarkov = emd.EigenmarkovDiffusion(
            n_particles=self.n_particles,
            n_spatial_locs=self.n_spatial_locs,
            n_time_pts=self.n_time_pts,
            impulse_idx=self.impulse_idx,
            scaling_factor=self.scaling_factor,
        )

        # run eigenmarkov simulation
        n_per_eigenmode_state = eigenmarkov.run_simulation(
            print_eigenvalues_and_vectors=False,
            print_init_conditions=False,
            print_transition_probability=False,
            plot_eigenvectors=False,
            plot_eigenmodes=False,
            plot_init_conditions=False,
            plot_simulation=False,
        )

        node_vals_from_modes = eigenmarkov.convert_to_spatial_nodes(
            n_per_eigenmode_state
        )

        if normalize:
            return node_vals_from_modes / self.n_particles
        else:
            return node_vals_from_modes

    def run_multi(self, normalize=False, make_dir=True, data_dir=None):
        if make_dir:
            from datetime import datetime

            time_now = datetime.now()  # UNIX time
            time_stamp = time_now.strftime("%Y%m%d_%H%M%S")

            if data_dir:
                eme_dir = r"{}/eme-validation/markov-eme/{}/".format(
                    data_dir, time_stamp
                )
            else:
                eme_dir = r"../../data/eme-validation/markov-eme/{}/".format(time_stamp)
            os.mkdir(eme_dir)
            print("Made new directory:", eme_dir)

        for i in range(self.n_runs):
            if i % 10 == 0:  # print once every 10 sims
                print("RUNNING SIMULATION {}".format(i))

            # run simulation
            node_vals_from_modes = self.run(normalize)

            # save output to csv
            np.savetxt(
                eme_dir + "/eme-run-{}.csv".format(f"{i:03}"),
                node_vals_from_modes,
                delimiter=",",
            )

        return eme_dir

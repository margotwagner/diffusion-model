import models.RandomWalk as rw
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
        particle_start_loc,
        scaling_factor=2,
    ):
        self.scaling_factor = scaling_factor
        self.n_runs = n_runs
        self.n_particles = n_particles
        self.n_spatial_locs = n_spatial_locs
        self.n_time_pts = n_time_pts
        self.particle_start_loc = particle_start_loc

    def run(self):

        eigenmarkov = emd.EigenmarkovDiffusion(
            n_particles=self.n_particles,
            n_spatial_locs=self.n_spatial_locs,
            n_time_pts=self.n_time_pts,
            particle_start_loc=self.particle_start_loc,
            scaling_factor=self.scaling_factor,
        )

        # run eigenmarkov simulation
        n_per_eigenmode_state = eigenmarkov.run_simulation()

        node_vals_from_modes = eigenmarkov.convert_to_spatial_nodes(
            n_per_eigenmode_state, print_output=False
        )

        return node_vals_from_modes

    def run_multi(self, make_dir=True):
        if make_dir:
            from datetime import datetime

            time_now = datetime.now()  # UNIX time
            time_stamp = time_now.strftime("%Y%m%d_%H%M%S")

            eme_dir = r"../data/eme-validation/markov-eme/{}/".format(time_stamp)
            os.makedirs(eme_dir)
            print("Made new directory:", eme_dir)

        for i in range(self.n_runs):
            if i % 10 == 0:  # print once every 10 sims
                print("RUNNING SIMULATION {}".format(i))

            # run simulation
            node_vals_from_modes = self.run()

            # save output to csv
            np.savetxt(
                eme_dir + "/eme-run-{}.csv".format(f"{i:03}"),
                node_vals_from_modes,
                delimiter=",",
            )
        
        return time_stamp

class RWRunMultiruns:
    def __init__(
        self,
        n_runs,
        n_particles,
        n_spatial_locs,
        n_time_pts,
        particle_start_loc,
    ):
        self.n_runs = n_runs
        self.n_particles = n_particles
        self.n_spatial_locs = n_spatial_locs
        self.n_time_pts = n_time_pts
        self.particle_start_loc = particle_start_loc

    def run(self):
        random_walk = rw.RandomWalk(
            n_particles=self.n_particles,
            n_spatial_locs=self.n_spatial_locs,
            n_time_pts=self.n_time_pts,
            particle_start_loc=self.particle_start_loc,
        )

        # run random walk simulation and plot output
        particle_locs = random_walk.run_simulation()

        unnorm_n_per_loc, n_per_loc, mean_n_per_loc = random_walk.postprocess_run(
            particle_locs, plot=False
        )

        return n_per_loc


    def run_multi(self, make_dir=True):
        if make_dir:
            from datetime import datetime

            time_now = datetime.now()  # UNIX time
            time_stamp = time_now.strftime("%Y%m%d_%H%M%S")

            rw_dir = r"../data/eme-validation/random-walk/{}/".format(time_stamp)
            os.makedirs(rw_dir)
            print("Made new directory:", rw_dir)

        # TODO: make option to add to existing dir

        for i in range(self.n_runs):
            if i % 10 == 0:  # print once every 10 sims
                print("RUNNING SIMULATION {}".format(i))

            # run simulation
            n_per_loc = self.run()

            # save output to csv
            np.savetxt(
                rw_dir + "/rw-run-{}.csv".format(f"{i:03}"), n_per_loc, delimiter=","
            )
        return time_stamp




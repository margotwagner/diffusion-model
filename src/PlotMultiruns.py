import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


class PlotMultiRuns(object):
    def __init__(
        self,
        rw_dir,
        eme_dir,
        n_runs,
        n_particles,
        n_spatial_locs,
        n_time_pts,
        particle_start_loc,
        plot_rw=True,
        plot_eme=True,
        line_length=4,
    ):
        self.rw_dir = rw_dir
        self.eme_dir = eme_dir
        self.n_runs = n_runs
        self.n_particles = n_particles
        self.n_spatial_locs = n_spatial_locs
        self.n_time_pts = n_time_pts
        self.particle_start_loc = particle_start_loc
        self.line_length = line_length
        self.plot_rw = plot_rw
        self.plot_eme = plot_eme

    def combine_runs(self, data_dir):
        # initialize array to store all runs
        runs = np.zeros((self.n_runs, self.n_spatial_locs, self.n_time_pts))

        # loop through runs
        for i in range(self.n_runs):
            # store run in array
            runs[i, :, :] = np.loadtxt(data_dir.format(f"{i:03}"), delimiter=",")

        # return array of all runs
        return runs

    def get_stats(self, data_dir):
        # combine runs
        runs = self.combine_runs(data_dir)

        # get mean and std
        mean = np.mean(runs, axis=0)
        std = np.std(runs, axis=0)

        # return mean and std
        return mean, std, runs

    def plot_mean(self, mean, colors):
        for i in range(self.n_spatial_locs):
            plt.plot(list(range(self.n_time_pts)), mean[i, :], color=colors[i], label=i)

    def plot_std(self, mean, std, colors):
        for i in range(self.n_spatial_locs):
            plt.fill_between(
                list(range(self.n_time_pts)),
                mean[i, :] + std[i, :],
                mean[i, :] - std[i, :],
                alpha=0.2,
                color=colors[i],
            )

    def plot_multiruns(self):
        plt.figure(figsize=(14, 10))

        # get list of colors
        colors = plt.cm.tab10_r(np.linspace(0, 1, self.n_spatial_locs))
        print(self.n_spatial_locs)

        if self.plot_rw:
            print("Preparing to plot random walk data...")

            # get data
            rw_mean, rw_std, rw_runs = self.get_stats(self.rw_dir)

            print("Plotting random walk data...")
            # plot mean
            self.plot_mean(rw_mean, colors)

            # plot std
            self.plot_std(rw_mean, rw_std, colors)

        if self.plot_eme:
            print("Preparing to plot eigenmarkov data...")

            # get data
            eme_mean, eme_std, eme_runs = self.get_stats(self.eme_dir)
            print(eme_runs.shape)
            print(eme_mean.shape)
            print(eme_std.shape)

            print("Plotting eigenmarkov data...")
            # plot mean
            self.plot_mean(eme_mean, colors)

            # plot std
            # self.plot_std(eme_mean, eme_std, colors)

        print("Beautifying plot...")
        plt.title(
            "Normalized number of particles in each position over time",
            fontsize=20,
        )
        plt.xlabel("timepoint", fontsize=14)
        plt.ylabel("normalized count", fontsize=14)
        plt.legend()
        plt.show()

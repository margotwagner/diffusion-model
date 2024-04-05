import numpy as np
import matplotlib.pyplot as plt
import glob


class RWPlotMultiRuns(object):
    def __init__(self, dir, file_id=None):
        self.dir = dir
        self.line_length = 4
        self.n_runs = self.n_runs()
        self.n_spatial_locs = self.n_spatial_locs()
        self.n_time_pts = self.n_time_pts()
        self.particle_start_loc = self.particle_start_loc()
        self.n_particles = self.n_particles()

    @property
    def spatial_mesh(self):
        """Return spatial mesh."""
        return np.linspace(0, self.line_length, self.n_spatial_locs)

    @property
    def time_mesh(self):
        return list(range(self.n_time_pts))

    def n_runs(self):
        n_runs = len(glob.glob(self.dir + "*"))
        return n_runs

    def n_spatial_locs(self):
        dir = glob.glob(self.dir + "*")[0]
        run = np.loadtxt(dir, delimiter=",")
        return run.shape[0]

    def n_time_pts(self):
        dir = glob.glob(self.dir + "*")[0]
        run = np.loadtxt(dir, delimiter=",")
        return run.shape[1]

    def particle_start_loc(self):
        dir = glob.glob(self.dir + "*")[0]
        run = np.loadtxt(dir, delimiter=",")
        start_loc = np.nonzero(run[:, 0])[0][0]

        return start_loc

    def n_particles(self):
        dir = glob.glob(self.dir + "*")[0]
        run = np.loadtxt(dir, delimiter=",")
        n_particles = run[self.particle_start_loc, 0]

        return n_particles

    def combine_runs(self):
        # initialize array to store all runs
        runs = np.zeros((self.n_runs, self.n_spatial_locs, self.n_time_pts))

        # loop through runs
        for i in range(self.n_runs):
            # store run in array
            runs[i, :, :] = np.loadtxt(f"{self.dir}rw-run-{i:03}.csv", delimiter=",")

        # return array of all runs
        return runs

    def get_stats(self, normalize=False):
        # combine runs
        runs = self.combine_runs()

        if normalize:
            runs = runs / self.n_particles

        # get mean and std
        mean = np.mean(runs, axis=0)
        std = np.std(runs, axis=0)

        # return mean and std
        return mean, std, runs

    def plot_mean_time(self, mean, time):
        if isinstance(time, int):
            plt.plot(self.spatial_mesh, mean[:, time], label=f"t = {time}")
        elif isinstance(time, list):
            time.reverse()
            for i in time:
                plt.plot(self.spatial_mesh, mean[:, i], label=f"t = {i}")

    def plot_mean_space(self, mean, space):
        if isinstance(space, int):
            plt.plot(
                self.time_mesh,
                np.transpose(mean[space, :]),
                label=f"$\Delta$x = {space - self.particle_start_loc + 1}",
            )
        elif isinstance(space, list):
            for i in space:
                plt.plot(
                    self.time_mesh,
                    np.transpose(mean[i, :]),
                    label=f"$\Delta$x = {i - self.particle_start_loc}",
                )

    def plot_std_time(self, mean, std, time):
        if isinstance(time, int):
            plt.fill_between(
                self.spatial_mesh,
                mean[:, time] + std[:, time],
                mean[:, time] - std[:, time],
                alpha=0.2,
            )
        elif isinstance(time, list):
            time.reverse()
            for i in time:
                plt.fill_between(
                    self.spatial_mesh,
                    mean[:, i] + std[:, i],
                    mean[:, i] - std[:, i],
                    alpha=0.2,
                )

    def plot_std_space(self, mean, std, space):
        if isinstance(space, int):
            plt.fill_between(
                self.time_mesh,
                mean[space, :] + std[space, :],
                mean[space, :] - std[space, :],
                alpha=0.2,
            )
        elif isinstance(space, list):
            for i in space:
                plt.fill_between(
                    self.time_mesh,
                    mean[i, :] + std[i, :],
                    mean[i, :] - std[i, :],
                    alpha=0.2,
                )

    def plot_multiruns_time(self, time):
        time.reverse()

        plt.figure(figsize=(14, 10))

        # get list of colors
        colors = plt.cm.tab10_r(np.linspace(0, 1, len(time)))

        print("Preparing to plot simulation data...")

        # get data
        mean, std, _ = self.get_stats(normalize=True)

        print("Plotting simulation data...")
        # plot mean
        self.plot_mean_time(mean, time)

        # plot std
        self.plot_std_time(mean, std, time)

        print("Beautifying plot...")
        plt.title(
            "Normalized number of particles in each position over time",
            fontsize=20,
        )
        plt.xlabel("distance (um)", fontsize=14)
        plt.ylabel("normalized count", fontsize=14)
        # plt.xlim([1.5, 3])
        plt.legend()
        plt.show()

    def plot_multiruns_space(self):
        space = [i + self.particle_start_loc for i in range(10)]

        plt.figure(figsize=(14, 10))

        # get list of colors
        colors = plt.cm.tab10_r(np.linspace(0, 1, len(space)))

        print("Preparing to plot simulation data...")

        # get data
        mean, std, _ = self.get_stats(normalize=True)

        print("Plotting simulation data...")
        # plot mean
        self.plot_mean_space(mean, space)

        # plot std
        self.plot_std_space(mean, std, space)

        print("Beautifying plot...")
        plt.title(
            "Normalized number of particles at each time over space",
            fontsize=20,
        )
        plt.xlabel("time (usec)", fontsize=14)
        plt.ylabel("normalized count", fontsize=14)
        plt.legend()
        plt.show()

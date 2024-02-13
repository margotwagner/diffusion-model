import numpy as np
import matplotlib.pyplot as plt
import glob


class PlotMultiRuns(object):
    def __init__(
        self,
        rw_dir="",
        eme_dir="",
        plot_rw=False,
        plot_eme=False,
    ):
        self.rw_dir = rw_dir
        self.eme_dir = eme_dir
        self.plot_rw = plot_rw
        self.plot_eme = plot_eme
        self.line_length = 4
        self.n_runs_rw = self.n_runs_rw()
        self.n_runs_eme = self.n_runs_eme()
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

    def n_runs_rw(self):
        n_runs_rw = len(glob.glob(self.rw_dir + "*"))
        return n_runs_rw

    def n_runs_eme(self):
        n_runs_eme = len(glob.glob(self.eme_dir + "*"))
        return n_runs_eme

    def n_spatial_locs(self):
        dir = glob.glob(self.rw_dir + "*")[0]
        run = np.loadtxt(dir, delimiter=",")
        return run.shape[0]

    def n_time_pts(self):
        dir = glob.glob(self.rw_dir + "*")[0]
        run = np.loadtxt(dir, delimiter=",")
        return run.shape[1]

    def particle_start_loc(self):
        dir = glob.glob(self.rw_dir + "*")[0]
        run = np.loadtxt(dir, delimiter=",")
        start_loc = np.nonzero(run[:, 0])[0][0]

        return start_loc

    def n_particles(self):
        dir = glob.glob(self.rw_dir + "*")[0]
        run = np.loadtxt(dir, delimiter=",")
        n_particles = run[self.particle_start_loc, 0]

        return n_particles

    def combine_runs(self, run_type):
        if run_type == "eme":
            data_dir = self.eme_dir
            n_runs = self.n_runs_eme
        else:
            data_dir = self.rw_dir
            n_runs = self.n_runs_rw

        # initialize array to store all runs
        runs = np.zeros((n_runs, self.n_spatial_locs, self.n_time_pts))

        # loop through runs
        for i in range(n_runs):
            # store run in array
            print(f"{data_dir}{run_type}-run-{i:03}.csv")
            runs[i, :, :] = np.loadtxt(
                f"{data_dir}{run_type}-run-{i:03}.csv", delimiter=","
            )

        # return array of all runs
        return runs

    def get_stats(self, run_type, normalize=False):
        # combine runs
        runs = self.combine_runs(run_type)

        if normalize:
            runs = runs / self.n_particles

        # get mean and std
        mean = np.mean(runs, axis=0)
        std = np.std(runs, axis=0)

        # return mean and std
        return mean, std, runs

    def plot_mean_time(self, mean, time, colors):
        # TODO: add colors
        if isinstance(time, int):
            plt.plot(self.spatial_mesh, mean[:, time], label=f"t = {time}")
        elif isinstance(time, list):
            time.reverse()
            for i in time:
                plt.plot(self.spatial_mesh, mean[:, i], label=f"t = {i}")

    def plot_mean_space(self, mean, space, colors):
        # TODO: add colors
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

    def plot_mean(self, mean, colors):
        for i in range(self.n_spatial_locs):
            plt.plot(list(range(self.n_time_pts)), mean[i, :], color=colors[i], label=i)

    def plot_mean_sep(self, mean, colors, shape, axs, label="test"):
        loc = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                if loc < self.n_spatial_locs:
                    axs[i, j].plot(
                        list(range(self.n_time_pts)),
                        mean[loc, :],
                        color=colors[loc],
                        label=label,
                    )
                    axs[i, j].axhline(color="black", linewidth=0.5, linestyle="--")
                    axs[i, j].set_title("Node {}".format(loc), fontsize=14)
                    loc += 1

    def plot_std_time(self, mean, std, time, colors):
        if isinstance(time, int):
            plt.fill_between(
                self.spatial_mesh,
                mean[:, time] + std[:, time],
                mean[:, time] - std[:, time],
                alpha=0.2,
                # color=colors[time],
            )
        elif isinstance(time, list):
            time.reverse()
            for i in time:
                plt.fill_between(
                    self.spatial_mesh,
                    mean[:, i] + std[:, i],
                    mean[:, i] - std[:, i],
                    alpha=0.2,
                    # color=colors[i],
                )

    def plot_std_space(self, mean, std, space, colors):
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

    def plot_std_sep(self, mean, std, colors, shape, axs):
        loc = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                if loc < self.n_spatial_locs:
                    axs[i, j].fill_between(
                        list(range(self.n_time_pts)),
                        mean[loc, :] + std[loc, :],
                        mean[loc, :] - std[loc, :],
                        alpha=0.2,
                        color=colors[i],
                    )
                    loc += 1

    def plot_std(self, mean, std, colors):
        for i in range(self.n_spatial_locs):
            plt.fill_between(
                list(range(self.n_time_pts)),
                mean[i, :] + std[i, :],
                mean[i, :] - std[i, :],
                alpha=0.2,
                color=colors[i],
            )

    def plot_multiruns_time(self, time):
        time.reverse()

        plt.figure(figsize=(14, 10))

        # get list of colors
        colors = plt.cm.tab10_r(np.linspace(0, 1, len(time)))

        if self.plot_eme:
            print("Preparing to plot eigenmarkov data...")

            # get data
            eme_mean, eme_std, _ = self.get_stats("eme", normalize=True)

            print("Plotting eigenmarkov data...")
            # plot mean
            self.plot_mean_time(eme_mean, time, colors)
            print(time)

            # plot std
            self.plot_std_time(eme_mean, eme_std, time, colors)

        if self.plot_rw:
            print("Preparing to plot random walk data...")

            # get data
            rw_mean, rw_std, _ = self.get_stats("rw", normalize=True)

            print("Plotting random walk data...")
            # plot mean
            self.plot_mean_time(rw_mean, time, colors)
            print(time)

            # plot std
            self.plot_std_time(rw_mean, rw_std, time, colors)

        print("Beautifying plot...")
        plt.title(
            "Normalized number of particles in each position over time",
            fontsize=20,
        )
        plt.xlabel("distance (um)", fontsize=14)
        plt.ylabel("normalized count", fontsize=14)
        plt.xlim([1.5, 3])
        plt.legend()
        plt.show()

    def plot_multiruns_space(self):
        space = [i + self.particle_start_loc for i in range(10)]

        plt.figure(figsize=(14, 10))

        # get list of colors
        colors = plt.cm.tab10_r(np.linspace(0, 1, len(space)))

        if self.plot_eme:
            print("Preparing to plot eigenmarkov data...")

            # get data
            eme_mean, eme_std, _ = self.get_stats("eme", normalize=True)

            print("Plotting eigenmarkov data...")
            # plot mean
            self.plot_mean_space(eme_mean, space, colors)

            # plot std
            self.plot_std_space(eme_mean, eme_std, space, colors)

        if self.plot_rw:
            print("Preparing to plot random walk data...")

            # get data
            rw_mean, rw_std, _ = self.get_stats("rw", normalize=True)

            print("Plotting random walk data...")
            # plot mean
            self.plot_mean_space(rw_mean, space, colors)

            # plot std
            self.plot_std_space(rw_mean, rw_std, space, colors)

        print("Beautifying plot...")
        plt.title(
            "Normalized number of particles at each time over space",
            fontsize=20,
        )
        plt.xlabel("time (usec)", fontsize=14)
        plt.ylabel("normalized count", fontsize=14)
        # plt.xlim([1.5, 3])
        plt.legend()
        plt.show()

    def plot_multiruns(self):
        plt.figure(figsize=(14, 10))

        # get list of colors
        colors = plt.cm.tab10_r(np.linspace(0, 1, self.n_spatial_locs))

        if self.plot_rw:
            print("Preparing to plot random walk data...")

            # get data
            rw_mean, rw_std, rw_runs = self.get_stats("rw", normalize=True)

            print("Plotting random walk data...")
            # plot mean
            self.plot_mean(rw_mean, colors)

            # plot std
            self.plot_std(rw_mean, rw_std, colors)

        if self.plot_eme:
            print("Preparing to plot eigenmarkov data...")

            # get data
            eme_mean, eme_std, eme_runs = self.get_stats("eme", normalize=True)

            print("Plotting eigenmarkov data...")
            # plot mean
            self.plot_mean(eme_mean, colors)

            # plot std
            self.plot_std(eme_mean, eme_std, colors)

        print("Beautifying plot...")
        plt.title(
            "Normalized number of particles in each position over time",
            fontsize=20,
        )
        plt.xlabel("timepoint", fontsize=14)
        plt.ylabel("normalized count", fontsize=14)
        # plt.legend()
        plt.show()

    def plot_separately(self, shape=(4, 3)):
        fig, axs = plt.subplots(shape[0], shape[1], figsize=(14, 10))

        # get list of colors
        colors = plt.cm.tab10_r(np.linspace(0, 1, self.n_spatial_locs))

        if self.plot_rw:
            print("Preparing to plot random walk data...")

            # get data
            rw_mean, rw_std, rw_runs = self.get_stats("rw")

            print("Plotting random walk data...")
            # plot mean
            self.plot_mean_sep(rw_mean, colors, shape, axs)

            # plot std
            self.plot_std_sep(rw_mean, rw_std, colors, shape, axs)

        if self.plot_eme:
            print("Preparing to plot eigenmarkov data...")

            # get data
            eme_mean, eme_std, eme_runs = self.get_stats("eme", normalize=True)

            print("Plotting eigenmarkov data...")
            # plot mean
            self.plot_mean_sep(eme_mean, colors, shape, axs)

            # plot std
            self.plot_std_sep(eme_mean, eme_std, colors, shape, axs)

        print("Beautifying plot...")
        fig.suptitle(
            "Normalized number of particles in each position over time",
            fontsize=20,
        )
        plt.xlabel("timepoint", fontsize=14)
        plt.ylabel("normalized count", fontsize=14)
        plt.tight_layout()
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib import rcParams


class ValidationPlots(object):
    def __init__(self, eme_dir, rw_dir, impulse_idx, n_particles_impulse):
        self.eme_dir = eme_dir
        self.rw_dir = rw_dir
        self.particle_start_loc = impulse_idx
        self.n_particles = n_particles_impulse
        self.line_length = 4
        self.n_runs = self.n_runs()
        self.n_spatial_locs = self.n_spatial_locs()
        self.n_time_pts = self.n_time_pts()

    @property
    def spatial_mesh(self):
        """Return spatial mesh."""
        return np.linspace(0, self.line_length, self.n_spatial_locs)

    @property
    def time_mesh(self):
        return list(range(self.n_time_pts))

    def n_runs(self):
        n_eme_runs = len(glob.glob(self.eme_dir + "*"))
        n_rw_runs = len(glob.glob(self.rw_dir + "*"))

        if n_eme_runs != n_rw_runs:
            raise ValueError(
                "Number of runs in EME and RW directories do not match. Exiting program."
            )

            exit()

        return n_eme_runs

    def n_spatial_locs(self):
        temp_eme_dir = glob.glob(self.eme_dir + "*")[0]
        eme_runs = np.loadtxt(temp_eme_dir, delimiter=",")
        n_spatial_locs_eme = eme_runs.shape[0]

        temp_rw_dir = glob.glob(self.rw_dir + "*")[0]
        rw_runs = np.loadtxt(temp_rw_dir, delimiter=",")
        n_spatial_locs_rw = rw_runs.shape[0]

        if n_spatial_locs_eme != n_spatial_locs_rw:
            raise ValueError(
                "Number of spatial locations in EME and RW directories do not match. Exiting program."
            )

            exit()

        return n_spatial_locs_eme

    def n_time_pts(self):
        temp_eme_dir = glob.glob(self.eme_dir + "*")[0]
        eme_run = np.loadtxt(temp_eme_dir, delimiter=",")
        n_time_pts_eme = eme_run.shape[1]

        temp_rw_dir = glob.glob(self.rw_dir + "*")[0]
        rw_run = np.loadtxt(temp_rw_dir, delimiter=",")
        n_time_pts_rw = rw_run.shape[1]

        if n_time_pts_eme != n_time_pts_rw:
            raise ValueError(
                "Number of time points in EME and RW directories do not match. Exiting program."
            )

            exit()

        return n_time_pts_eme

    def combine_runs(self, type):
        # initialize array to store all runs
        runs = np.zeros((self.n_runs, self.n_spatial_locs, self.n_time_pts))

        if type == "eme":
            dir = self.eme_dir
        elif type == "rw":
            dir = self.rw_dir
        else:
            raise ValueError("Invalid type. Exiting program.")

            exit()

        # loop through runs
        for i in range(self.n_runs):
            runs[i, :, :] = np.loadtxt(f"{dir}{type}-run-{i:03}.csv", delimiter=",")

        # return array of all runs
        return runs

    def get_stats(self, type, normalize=False):
        # combine runs
        runs = self.combine_runs(type)

        if normalize:
            runs = runs / self.n_particles

        # get mean and std
        mean = np.mean(runs, axis=0)
        std = np.std(runs, axis=0)

        # return mean and std
        return mean, std, runs

    def prep_eme_data(self, abs=False):
        mean_eme, _, _ = self.get_stats(type="eme", normalize=False)
        mean_eme = mean_eme / np.max(mean_eme)

        if abs:
            mean_eme = np.abs(mean_eme)

        return mean_eme

    def percent_difference(self, mean_1, mean_2):
        return np.abs(mean_1 - mean_2) / ((mean_1 + mean_2) / 2)

    def percent_error(self, mean_1, mean_2):
        return np.abs(mean_1 - mean_2) / mean_1

    def plot_eme_3d(self, abs=False, cmap="veridis"):
        rcParams["xtick.color"] = "white"
        rcParams["ytick.color"] = "white"
        rcParams["axes.labelcolor"] = "white"
        rcParams["axes.edgecolor"] = "white"
        fig = plt.figure(figsize=(10, 10), dpi=125)
        ax = plt.axes(projection="3d")

        X, Y = np.meshgrid(self.time_mesh, self.spatial_mesh)

        mean_eme = self.prep_eme_data(abs)

        ax.plot_surface(
            X,
            Y,
            mean_eme,
            rstride=1,
            cstride=1,
            cmap=cmap,
            edgecolor="none",
        )

        # ax.set_title("EigenMarkov Diffusion", fontsize=20)
        ax.set_xlabel("time", fontsize=24, color="white", labelpad=20)
        ax.set_ylabel("space", fontsize=24, color="white", labelpad=20)
        ax.set_zlabel("particle count", fontsize=24, color="white", labelpad=20)
        ax.tick_params(axis="both", which="major", labelsize=20, color="white")
        plt.savefig(
            "/Users/margotwagner/projects/diffusion-model/figures/eme_3d.svg",
            transparent=True,
        )
        plt.show()

    def plot_rw_3d(self):
        fig = plt.figure(figsize=(10, 10), dpi=125)
        ax = plt.axes(projection="3d")

        X, Y = np.meshgrid(self.time_mesh, self.spatial_mesh)

        mean_rw, _, _ = self.get_stats(type="rw", normalize=True)

        ax.plot_surface(
            X,
            Y,
            mean_rw,
            rstride=1,
            cstride=1,
            cmap="viridis",
            edgecolor="none",
        )

        ax.set_title("Random Walk Diffusion", fontsize=16)
        ax.set_xlabel("time", fontsize=12)
        ax.set_ylabel("space", fontsize=12)
        ax.set_zlabel("particle count", fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=12)
        plt.show()

    def plot_percent_difference_3d(self, abs=False):
        fig = plt.figure(figsize=(16, 10), dpi=125)
        ax = plt.axes(projection="3d")

        X, Y = np.meshgrid(self.time_mesh, self.spatial_mesh)  # X, Y

        mean_rw, _, _ = self.get_stats(type="rw", normalize=True)
        mean_eme = self.prep_eme_data(abs)

        ax.plot_surface(
            X,
            Y,
            self.percent_difference(mean_rw, mean_eme),
            rstride=1,
            cstride=1,
            cmap="viridis",
            edgecolor="none",
        )

        # ax.set_title("Percent Difference", fontsize=20)
        ax.set_xlabel("time", fontsize=24, labelpad=10)
        ax.set_ylabel("space", fontsize=24, labelpad=10)
        # ax.set_zlabel("percentage (%)", fontsize=24, labelpad=10)
        ax.tick_params(axis="both", which="major", labelsize=20)
        plt.tight_layout()
        plt.show()

    def plot_percent_error_3d(self, abs=False):
        fig = plt.figure(figsize=(10, 10), dpi=125)
        ax = plt.axes(projection="3d")

        X, Y = np.meshgrid(self.time_mesh, self.spatial_mesh)  # X, Y

        mean_rw, _, _ = self.get_stats(type="rw", normalize=True)
        mean_eme = self.prep_eme_data(abs)

        ax.plot_surface(
            X,
            Y,
            self.percent_error(mean_rw, mean_eme),
            rstride=1,
            cstride=1,
            cmap="viridis",
            edgecolor="none",
        )

        ax.set_title("Percent Error", fontsize=16)
        ax.set_xlabel("time", fontsize=12)
        ax.set_ylabel("space", fontsize=12)
        ax.set_zlabel("percentage (%)", fontsize=12)
        ax.tick_params(axis="both", which="major", labelsize=12)
        plt.show()

    def plot_difference_3d(self, abs=False):
        fig = plt.figure(figsize=(10, 10), dpi=125)
        ax = plt.axes(projection="3d")

        X, Y = np.meshgrid(self.time_mesh, self.spatial_mesh)  # X, Y

        mean_rw, _, _ = self.get_stats(type="rw", normalize=True)
        mean_eme = self.prep_eme_data(abs)

        ax.plot_surface(
            X,
            Y,
            mean_eme - mean_rw,
            rstride=1,
            cstride=1,
            cmap="viridis",
            edgecolor="none",
        )
        # ax.set_title("Difference", fontsize=20)
        ax.set_xlabel("time", fontsize=24, labelpad=10)
        ax.set_ylabel("space", fontsize=24, labelpad=10)
        # ax.set_zlabel("particle count", fontsize=24)
        ax.tick_params(axis="both", which="major", labelsize=20)
        plt.tight_layout()
        plt.show()

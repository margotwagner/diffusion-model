import numpy as np
import matplotlib.pyplot as plt
import glob as glob

class PlotMultiRuns(object):
    def __init__(self, dir, file_id=None, eme_particles=50):
        self.dir = dir
        self.file_id = file_id
        self.line_length = 4
        self.n_runs = self.n_runs()
        self.n_spatial_locs = self.n_spatial_locs()
        self.n_time_pts = self.n_time_pts()
        self.particle_start_loc = self.particle_start_loc()
        self.eme_particles = eme_particles
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
        print("n_runs:", n_runs)
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
        print("start_loc:", start_loc)
        return start_loc

    def n_particles(self):
        dir = glob.glob(self.dir + "*")[0]
        run = np.loadtxt(dir, delimiter=",")
        n_particles = run[self.particle_start_loc, 0]
        print("n_particles:", n_particles)
        if self.file_id == "eme":
            return self.eme_particles
        return n_particles

    def combine_runs(self):
        # initialize array to store all runs
        runs = np.zeros((self.n_runs, self.n_spatial_locs, self.n_time_pts))
        # loop through runs
        for i in range(self.n_runs):
            # store run in array
            # print(f"{data_dir}{run_type}-run-{i:03}.csv")
            runs[i, :, :] = np.loadtxt(
                f"{self.dir}{self.file_id}-run-{i:03}.csv", delimiter=","
            )

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

    def plot_mean_time(self, mean, time, axis = plt):
        if isinstance(time, int):
            axis.plot(self.spatial_mesh, mean[:, time], label=f"t = {time}")
        elif isinstance(time, list):
            time.reverse()
            for i in time:
                axis.plot(self.spatial_mesh, mean[:, i], label=f"t = {i}")

    def plot_mean_space(self, mean, space, axis = plt):
        if isinstance(space, int):
            axis.plot(
                self.time_mesh,
                np.transpose(mean[space, :]),
                label=f"$\Delta$x = {space - self.particle_start_loc + 1}",
            )
        elif isinstance(space, list):
            for i in space:
                axis.plot(
                    self.time_mesh,
                    np.transpose(mean[i, :]),
                    label=f"$\Delta$x = {i - self.particle_start_loc}",
                )

    def plot_std_time(self, mean, std, time, axis = plt):
        if isinstance(time, int):
            axis.fill_between(
                self.spatial_mesh,
                mean[:, time] + std[:, time],
                mean[:, time] - std[:, time],
                alpha=0.2,
            )
        elif isinstance(time, list):
            time.reverse()
            for i in time:
                axis.fill_between(
                    self.spatial_mesh,
                    mean[:, i] + std[:, i],
                    mean[:, i] - std[:, i],
                    alpha=0.2,
                )

    def plot_std_space(self, mean, std, space, axis = plt):
        if isinstance(space, int):
            axis.fill_between(
                self.time_mesh,
                mean[space, :] + std[space, :],
                mean[space, :] - std[space, :],
                alpha=0.2,
            )
        elif isinstance(space, list):
            for i in space:
                axis.fill_between(
                    self.time_mesh,
                    mean[i, :] + std[i, :],
                    mean[i, :] - std[i, :],
                    alpha=0.2,
                )

    def plot_multiruns_time(self, time, axis = None):
        time.reverse()
        if axis == None:
            axis = plt
            plt.figure(figsize=(14, 10))

        # get list of colors
        colors = plt.cm.tab10_r(np.linspace(0, 1, len(time)))

        print("Preparing to plot simulation data...")
        mean, std = None, None
        # get data
        if self.file_id == "eme":
            mean, std, _ = self.get_stats(normalize=True)
        elif self.file_id == "rw":
            mean, std, _ = self.get_stats(normalize=True)
        else:
            raise ValueError("Invalid file_id. Please choose 'eme' or 'rw'.")

        print("Plotting simulation data...")
        # plot mean
        self.plot_mean_time(mean, time, axis=axis)

        # plot std
        self.plot_std_time(mean, std, time, axis=axis)

        print("Beautifying plot...")
        if axis != plt:
            axis.set_title(
                "Normalized number of particles in each position over time",
                fontsize=20,
            )
            axis.set_xlabel("distance (um)", fontsize=14)
            axis.set_ylabel("normalized count", fontsize=14)
            # plt.xlim([1.5, 3])
            axis.legend(title="timesteps")
        else:
            axis.title(
                "Normalized number of particles in each position over time",
                fontsize=20,
            )
            axis.xlabel("distance (um)", fontsize=14)
            axis.ylabel("normalized count", fontsize=14)
            # plt.xlim([1.5, 3])
            axis.legend(title="timesteps")
            axis.show()
        return axis

    def plot_multiruns_space(self, axis = None, steps_from_impulse=10):
        if axis == None:
            axis = plt
            plt.figure(figsize=(14, 10))
        space = [i + self.particle_start_loc for i in range(steps_from_impulse)]


        # get list of colors
        colors = plt.cm.tab10_r(np.linspace(0, 1, len(space)))

        print("Preparing to plot simulation data...")

        # get data
        if self.file_id == "eme":
            mean, std, _ = self.get_stats(normalize=True)
        elif self.file_id == "rw":
            mean, std, _ = self.get_stats(normalize=True)
        else:
            raise ValueError("Invalid file_id. Please choose 'eme' or 'rw'.")

        print("Plotting simulation data...")
        self.plot_mean_space(mean, space, axis=axis)
        self.plot_std_space(mean, std, space, axis=axis)

        print("Beautifying plot...")
        if axis != plt:
            axis.set_title(
                "Normalized number of particles at each time over space",
                fontsize=20,
            )
            axis.set_xlabel("time (usec)", fontsize=14)
            axis.set_ylabel("normalized count", fontsize=14)
            axis.legend(title="steps from impulse")
        else:
            axis.title(
                "Normalized number of particles at each time over space",
                fontsize=20,
            )
            axis.xlabel("time (usec)", fontsize=14)
            axis.ylabel("normalized count", fontsize=14)
            axis.legend(title="steps from impulse")
            axis.show()
        return axis

    def compare_overlap_space(self,compare, axis = None):
        # plot with time on the x-axis
        if axis == None:
            fig, axis = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
        x_idx = [compare.impulse_idx + i for i in range(0, 10)]
        x_labels = [*range(0, 10)]
        for i in range(len(x_idx)):
            axis.plot(
                compare.time_mesh,
                compare.u_diff[x_idx[i], :] / compare.n_ca,
                color="black"
            )
        return self.plot_multiruns_space(axis=axis)

    def compare_overlap_time(self, time, compare, axis = None):
        if axis == None:
            fig, axis = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
        for i in time:
            axis.plot(
                compare.spatial_mesh, compare.u_diff[:, i] / compare.n_ca, color="black"
            )

        return self.plot_multiruns_time(time=time, axis=axis)

    # HACKY SOLUTIONS
    def eme_bug_plot_multiruns_space(self, axis = None, steps_from_impulse=41, nonsense=False):
        if axis == None:
            axis = plt
            plt.figure(figsize=(14, 10))
        if nonsense:
            space = [i + self.particle_start_loc for i in range(steps_from_impulse)][:-10]
        else:
            space = [i + self.particle_start_loc for i in range(steps_from_impulse)][-10:]


        # get list of colors
        colors = plt.cm.tab10_r(np.linspace(0, 1, len(space)))

        print("Preparing to plot simulation data...")

        # get data
        if self.file_id == "eme":
            mean, std, _ = self.get_stats(normalize=True)
        elif self.file_id == "rw":
            mean, std, _ = self.get_stats(normalize=True)
        else:
            raise ValueError("Invalid file_id. Please choose 'eme' or 'rw'.")

        print("Plotting simulation data...")
        self.plot_mean_space(mean, space, axis=axis)
        self.plot_std_space(mean, std, space, axis=axis)

        print("Beautifying plot...")
        if axis != plt:
            axis.set_title(
                "Normalized number of particles at each time over space",
                fontsize=20,
            )
            axis.set_xlabel("time (usec)", fontsize=14)
            axis.set_ylabel("normalized count", fontsize=14)
            axis.legend(title="last 10 steps ")
        else:
            axis.title(
                "Normalized number of particles at each time over space",
                fontsize=20,
            )
            axis.xlabel("time (usec)", fontsize=14)
            axis.ylabel("normalized count", fontsize=14)
            axis.legend(title="last 10 steps")
            axis.show()
        return axis
    
    def eme_bug_plot_multiruns_time(self, time, axis = None):
        # NOT REVERSING TIME
        time.reverse()
        if axis == None:
            axis = plt
            plt.figure(figsize=(14, 10))

        # get list of colors
        colors = plt.cm.tab10_r(np.linspace(0, 1, len(time)))

        print("Preparing to plot simulation data...")
        mean, std = None, None
        # get data
        if self.file_id == "eme":
            mean, std, _ = self.get_stats(normalize=True)
        elif self.file_id == "rw":
            mean, std, _ = self.get_stats(normalize=True)
        else:
            raise ValueError("Invalid file_id. Please choose 'eme' or 'rw'.")

        print("Plotting simulation data...")
        # plot mean
        self.plot_mean_time(mean, time, axis=axis)

        # plot std
        self.plot_std_time(mean, std, time, axis=axis)

        print("Beautifying plot...")
        if axis != plt:
            axis.set_title(
                "Normalized number of particles in each position over time",
                fontsize=20,
            )
            axis.set_xlabel("distance (um)", fontsize=14)
            axis.set_ylabel("normalized count", fontsize=14)
            # plt.xlim([1.5, 3])
            axis.legend(title="timesteps")
        else:
            axis.title(
                "Normalized number of particles in each position over time",
                fontsize=20,
            )
            axis.xlabel("distance (um)", fontsize=14)
            axis.ylabel("normalized count", fontsize=14)
            # plt.xlim([1.5, 3])
            axis.legend(title="timesteps")
            axis.show()
        return axis

    def eme_bug_compare_overlap_space(self,compare, axis = None):
        # plot with time on the x-axis
        if axis == None:
            fig, axis = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
        x_idx = [compare.impulse_idx + i for i in range(0, 10)]
        x_labels = [*range(0, 10)]
        for i in range(len(x_idx)):
            axis.plot(
                compare.time_mesh,
                compare.u_diff[x_idx[i], :] / compare.n_ca,
                color="black"
            )
        return self.eme_bug_plot_multiruns_space(axis=axis)

    def eme_bug_compare_overlap_time(self, time, compare, axis = None):
        if axis == None:
            fig, axis = plt.subplots(1, 1, figsize=(10, 5), sharey=True)
        for i in time:
            axis.plot(
                compare.spatial_mesh, compare.u_diff[:, i] / compare.n_ca, color="black"
            )

        return self.eme_bug_plot_multiruns_time(time=time, axis=axis)

import numpy as np
from typing import Union, Tuple
import matplotlib.pyplot as plt


class RandomWalk:
    def __init__(
        self,
        n_particles: int,  # number of molecules
        n_spatial_locs: int,  # define number of grid points along 1D line
        n_time_pts: int,  # number of time points
        particle_start_loc: int,  # start position of input impulse molecules
        dt: Union[int, float] = 1,  # time step (usec)
        line_length: Union[
            int, float
        ] = 4,  # length of line on which molecule is diffusing (um)
        diffusion_constant_D: float = 2.20e-4,  # Calcium diffusion coeff (um^2/usec)
    ):
        self.n_spatial_locs = n_spatial_locs
        self.dt = dt
        self.n_time_pts = n_time_pts
        self.particle_start_loc = particle_start_loc
        self.line_length = line_length
        self.n_particles = n_particles
        self.diffusion_constant_D = diffusion_constant_D
        self.last_elem_i = -1
        self.min_x_pos = 0
        self.max_x_pos = self.n_spatial_locs - 1

    def get_jump_probability(
        self,
    ) -> Tuple[float, float]:
        """Find the probability of moving one spot to the left or right based on
        finite-difference approximations
        Rate constant, k = D/dx^2
        P(move one spot to the right/left) = k*dt

        return:
            - probability of diffusing one spot to the left or right (k*dt)
            - k: diffusion rate constant

        """

        dx = self.line_length / self.n_spatial_locs  # distance of one "hop"
        diffusion_rate_constant_k = self.diffusion_constant_D / dx**2  # rate constant

        return diffusion_rate_constant_k * self.dt, diffusion_rate_constant_k

    def run_simulation(
        self,
    ) -> np.ndarray:
        """1-D random walk for n_particles from a range of
        positions = [0, (n_spatial_locs - 1)]
        Implementation similar to GeeksforGeeks 'Random Walk (Implementation in Python)'

        Returns:
        positions of all particles over time - matrix shaped
            (n_particles, n_time_pts)
        """
        # get jump probability
        jump_probability = self.get_jump_probability()[0]

        # initialize array for all particle positions (number of particles)
        particle_locs = np.empty((self.n_particles, self.n_time_pts), dtype="int64")

        for n in range(self.n_particles):
            # Initialize starting position (0 to (n_spatial_locs - 1) range)
            positions = [self.particle_start_loc]

            # sampling probability all at once (1000 timepoints)
            rand = np.random.random(self.n_time_pts - 1)

            # movement decision conditions
            move_l_cond = rand < jump_probability
            move_r_cond = rand > (1 - jump_probability)
            # stay condition is between the two

            # run simulation for particle n
            # check probability rolls
            for move_left, move_right in zip(move_l_cond, move_r_cond):
                # move left if move_left=True and last position != minimum
                # position
                left = move_left and positions[self.last_elem_i] > self.min_x_pos

                # move right if move_right=True and last position != maximum
                # position
                right = move_right and positions[self.last_elem_i] < self.max_x_pos

                # stay condition is implied

                # adjust position accordingly
                positions.append(positions[self.last_elem_i] - left + right)

            # add results to cumulative array
            particle_locs[n] = positions

        return particle_locs

    def postprocess_run(
        self, particle_locs: np.ndarray, plot: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Post-process 1D random-walk diffusion
        (counts, normalized counts, means)

        Args:
        plot:
            option to plot particle counts for each position (default: False)

        Returns:
        unnorm_n_per_loc:
            number of particles in each position over time (unnormalized)
        n_per_loc:
            number of particles in each position over time (normalized)
        mean_n_per_loc:
            mean value of particle number at each location over the whole
            simulation
        """

        # distribution of particles across positions over time
        unnorm_n_per_loc = np.zeros(
            (self.n_spatial_locs, self.n_time_pts), dtype="int64"
        )  # number of particles over time

        n_per_loc = np.zeros(
            (self.n_spatial_locs, self.n_time_pts)
        )  # normalized count over time

        for i in range(self.n_time_pts):
            # count number of particles in each position
            counts = np.bincount(particle_locs[:, i])

            # resize to include all positions if it doesn't already
            counts.resize(self.n_spatial_locs)

            # assign number of particles
            unnorm_n_per_loc[:, i] = counts

            # normalize counts and assign
            counts = counts / particle_locs.shape[0]
            n_per_loc[:, i] = counts

        mean_n_per_loc = np.mean(unnorm_n_per_loc, axis=1)

        if plot:
            # plot particle counts for each position
            plt.figure(figsize=(14, 10))

            for i in range(self.n_spatial_locs):
                plt.plot(list(range(self.n_time_pts)), n_per_loc[i, :])

            plt.title(
                "Normalized number of particles in each position over time",
                fontsize=20,
            )
            plt.xlabel("timepoint", fontsize=14)
            plt.ylabel("normalized count", fontsize=14)
            plt.legend(list(range(self.n_spatial_locs)))
            plt.show()

        return unnorm_n_per_loc, n_per_loc, mean_n_per_loc

    def draw_impulse(self):
        """
        just for fun
        """
        left = self.particle_start_loc
        right = (
            self.n_spatial_locs - self.particle_start_loc
        ) - 1  # -1 for the impulse itself
        print("")
        print("~" * left, ".", "~" * right)
        print("_" * left, "|", "_" * right)
        print("-" * left, "|", "-" * right)
        print("=" * left, "|", "=" * right)
        print(
            " " * left,
            "^Impulse @ {} / {}".format(self.particle_start_loc, self.n_spatial_locs),
        )

    def get_variance(self):
        """Get the variance of the 1d random walk diffusion process. Definition from https://mathworld.wolfram.com/RandomWalk1-Dimensional.html"""

        jump_probability, _ = self.get_jump_probability()

        return self.n_particles * (jump_probability**2)

    def get_std(self):
        """Get the standard deviation of the 1d random walk diffusion process. Definition from https://mathworld.wolfram.com/RandomWalk1-Dimensional.html"""

        return np.sqrt(self.get_variance())

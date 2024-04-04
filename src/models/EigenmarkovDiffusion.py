"""Class for simulating calcium diffusion using stochastic Markov version of eigenmode-based diffusion.

Usage: initialize class with parameters, then run simulation with run_simulation() method. See run_validation.py for example usage.
"""

__author__ = ["Margot Wagner"]
__contact__ = "mwagner@ucsd.edu"
__date__ = "2023/06/13"

import numpy as np
from typing import Union, Tuple
from numpy.linalg import eig


class EigenmarkovDiffusion:
    def __init__(
        self,
        n_particles: int,  # number of molecules
        n_spatial_locs: int,  # define number of grid points along 1D line,
        n_time_pts: int,  # number of time points
        particle_start_loc: int,  # start position of input impulse molecules
        scaling_factor: float,  # scaling factor for mode <-> node mapping
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
        self.scaling_factor = scaling_factor
        self.line_length = line_length
        self.n_particles = n_particles
        self.diffusion_constant_D = diffusion_constant_D

    def get_jump_probability(
        self,
    ) -> Tuple[float, float]:
        """Find the probability of moving one spot to the left or right based on finite-difference approximations
        Rate constant, k = D/dx^2
        P(move one spot to the right/left) = k*dt

        return:
            float: probability of diffusing one spot to the left or right (k*dt)
            float: diffusion rate constant (k)

        """

        dx = self.line_length / self.n_spatial_locs  # distance of one "hop"
        diffusion_rate_constant_k = self.diffusion_constant_D / dx**2  # rate constant

        return diffusion_rate_constant_k * self.dt, diffusion_rate_constant_k

    def get_transition_matrix(self) -> np.ndarray:
        """Builds and returns the transition matrix for the 1D random walk case
        as given.

        returns:
            np.array: transition matrix
        """
        # get diffusion rate constant
        diffusion_rate_constant_k = self.get_jump_probability()[1]

        # Define A (transition) matrix
        A = np.zeros(
            (self.n_spatial_locs, self.n_spatial_locs)
        )  # transition probability between grid points

        # Transition matrix is given by the ODE dynamics equation (using k-values)
        vec_diag = np.full(self.n_spatial_locs, (2 * diffusion_rate_constant_k))
        vec_off_diag = np.full(
            (self.n_spatial_locs - 1), -diffusion_rate_constant_k
        )  # off-diagonal values

        # create transition matrix
        A = (
            np.diag(vec_diag, k=0)
            + np.diag(vec_off_diag, k=1)
            + np.diag(vec_off_diag, k=-1)
        )
        A[0, 0] = diffusion_rate_constant_k
        A[self.n_spatial_locs - 1, self.n_spatial_locs - 1] = diffusion_rate_constant_k

        return A

    def get_eigenvalues_and_vectors(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the sorted eigenvalues and eigenvectors of matrix A

        Returns:
            np.array: eigenvalues - 1d matrix of size n_spatial_locs
            np.array: eigenvectors - 2d matrix of eigenvectors where columns
                    correspond to eigenvalues (ie evec[:,k] <-> eval[k])
        """
        eigenvalues, eigenvectors = eig(self.get_transition_matrix())

        return eigenvalues, eigenvectors

    def get_eme_init_conditions(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find the initial normalized number of particles in the positive
        and negative state of eigenmode k

        params:
            eigenvectors:
                eigenvector columns correspond to eigenvalues
                (ie evec[:,k] <-> eval[k])
                eigenvector[e, eigenmode (k)]
            particle_start_loc:
                location of impulse

        return:
            normalized number of particles in each eigenmode at time = 0
        """
        _, eigenvectors = self.get_eigenvalues_and_vectors()

        print(f"PARTICLE STARTING LOCATION: {self.particle_start_loc}")
        start_loc_eigenvector = eigenvectors[self.particle_start_loc, :]

        # UNNORMALIZED SOLUTION
        n_per_positive_mode = 0.5 * (
            np.sqrt(self.n_particles**2 * start_loc_eigenvector**2)
            + (self.n_particles * start_loc_eigenvector)
        )

        n_per_negative_mode = 0.5 * (
            np.sqrt(self.n_particles**2 * start_loc_eigenvector**2)
            - (self.n_particles * start_loc_eigenvector)
        )

        return n_per_positive_mode, n_per_negative_mode

    def get_eigenmode_transition_probability(self) -> np.ndarray:
        """Get the probability of transitioning from one eigenmode to another

        Args:
            print_output (bool, optional): Whether to print outputs. Defaults to False.

        Returns:
            np.ndarray: transition probability
        """

        eigenvalues, _ = self.get_eigenvalues_and_vectors()
        transition_probability = (eigenvalues / 2) * self.dt

        return transition_probability

    def run_simulation(
        self,
        binomial_sampling=False,
    ) -> np.ndarray:
        """Markov simulation for eigenmode analysis to capture calcium diffusion
        over time

        params:
            binomial_sampling: whether to use binomial sampling or not      (default: False)

        return:
            n_per_eigenmode_state: normalized number of particles in each
            eigenmode (+/-) at each timepoint
            np array shape (n_modes x n_time x n_eigenmode_states)
        """

        # positive and negative states
        n_spins = 2

        # initialize number of particles
        # n_modes x n_time x n_spins (for +/-, this is 2)
        n_per_eigenmode_state = np.zeros(
            (self.n_spatial_locs, self.n_time_pts, n_spins)
        ).astype("int")

        # assign initial conditions using number of molecules
        init_cond = self.get_eme_init_conditions()
        init_cond = (
            np.rint(init_cond) / self.scaling_factor
        )  # round initial conditions to nearest int

        # get transition probability
        transition_probability = self.get_eigenmode_transition_probability()

        # initialize the number of particles in each eigenmode state
        for j in range(n_spins):
            n_per_eigenmode_state[:, 0, j] = init_cond[j]

        # for each time point
        for i in range(self.n_time_pts - 1):
            # for each eigenmode
            for k in range(self.n_spatial_locs):
                # initialize the number of particles that transition
                # [from + -> -, from - -> +]
                n_change = [0, 0]

                # find number of transitions positive/negative eigenmode state;
                for j in range(n_spins):
                    if binomial_sampling:
                        # sum number of particles that left current state given by binomial sampling
                        n_change[j] = np.random.binomial(
                            n_per_eigenmode_state[k, i, j], transition_probability[k]
                        )

                    else:
                        # sample random numbers equal to number of particles in
                        # current state
                        r = np.random.random(n_per_eigenmode_state[k, i, j])

                        # sum number of particles that left current state
                        n_change[j] = sum(r < transition_probability[k])

                # update next time point
                for j in range(n_spins):
                    n_per_eigenmode_state[k, i + 1, j] = (
                        n_per_eigenmode_state[k, i, j] - n_change[j] + n_change[1 - j]
                    )

        return n_per_eigenmode_state

    def convert_to_spatial_nodes(
        self,
        n_per_eigenmode_state: np.ndarray,
    ) -> np.ndarray:
        """Calculate the number of particles at each node from the eigenmode
        representation.

        Args:
            n_per_eigenmode_state: normalize the number of particles in each node;
            np aray shape (n_modes x n_time x n_eigenmode_states)
            eigenvectors: eigenvector of node i (vector); v[:,k] is the eigenvector
            corresponding to the eigenvalue w[k]; (ie evec[:,k] <-> eval[k])
                eigenvector[e, eigenmode (k)]


        Returns:
            np array containing normalized particle counts for each node
            (n_nodes x n_time_pts)
        """
        _, eigenvectors = self.get_eigenvalues_and_vectors()

        # initialize node values (n_nodes x n_time_pts)
        node_vals_from_modes = np.zeros((self.n_spatial_locs, self.n_time_pts))

        # positive - negative
        n_per_eigenmode = (
            n_per_eigenmode_state[:, :, 0] - n_per_eigenmode_state[:, :, 1]
        )

        # for each spatial node
        for i in range(self.n_spatial_locs):
            node_vals_from_modes[i, :] = (
                np.dot(eigenvectors[i, :], n_per_eigenmode) / self.n_spatial_locs
            )

        return self.scaling_factor * node_vals_from_modes

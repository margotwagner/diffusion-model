"""Class for simulating calcium diffusion using stochastic Markov version of eigenmode-based diffusion.

Usage: initialize class with parameters, then run simulation with run_simulation() method. See run_validation.py for example usage.
"""

__author__ = ["Margot Wagner"]
__contact__ = "mwagner@ucsd.edu"
__date__ = "2023/06/13"

import numpy as np
from typing import Union, Tuple
from numpy.linalg import eig
import matplotlib.pyplot as plt
import math


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

    def get_eigenmode(self, eigenvalues, t):
        """Returns the eigenmode for a given eigenvalue and time

        Args:
            eigenvalues (np.array): eigenvalues of the transition matrix
            t (_type_): time point(s) of interest

        Returns:
            _type_: eigenmode
        """
        return np.exp(-eigenvalues * t)

    def make_eigenmode_plots(self, eigenvalues):
        """Plots the eigenmodes for a given set of eigenvalues

        Args:
            eigenvalues (_type_): eigenvalues of the transition matrix
        """
        time = np.array(range(self.n_time_pts))
        alpha = 1  # transparency
        for λ in eigenvalues:
            plt.plot(
                time,
                self.get_eigenmode(λ, time),
                c="red",
                alpha=alpha,
                label="λ={:.4f}".format(λ),
            )

            alpha *= 0.66  # use transparency to generate a gradient in colors

        plt.xlabel("t [µs]")
        plt.ylabel("$e^{-λt}$")
        plt.legend(bbox_to_anchor=(1, 1))
        plt.show()

    def make_eigenvector_plots(self, eigenvalues, eigenvectors):
        """Plots the eigenvectors for a given set of eigenvalues

        Args:
            eigenvalues (np.array): eigenvalues of the transition matrix
            eigenvectors (np.array): eigenvectors of the transition matrix
        """

        num_nodes, num_modes = eigenvectors.shape
        alpha = 1
        for k in range(num_modes):
            λ = eigenvalues[k]
            v = eigenvectors[:, k]
            # print(v)

            plt.plot(v, "*-", c="red", alpha=alpha, label="λ={:.4f}".format(λ))

            alpha *= 0.66  # use transparency to generate a gradient in colors

        plt.xlabel("Spatial Node")
        plt.ylabel("Eigenvector")
        plt.legend(bbox_to_anchor=(1, 1))
        plt.show()

    def get_eigenvalues_and_vectors(
        self,
        print_output=False,
        plot_eigenmodes=False,
        plot_eigenvectors=False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the sorted eigenvalues and eigenvectors of matrix A

        Args:
            print_output (bool, optional): Whether to print eigenvalues and vectors (default: False)
            plot_eigenmodes (bool, optional): Whether to plot eigenmodes (default: False)
            plot_eigenvectors (bool, optional): Whether to plot eigenvectors (default: False)

        Returns:
            np.array: eigenvalues - 1d matrix of size n_spatial_locs
            np.array: eigenvectors - 2d matrix of eigenvectors where columns
                    correspond to eigenvalues (ie evec[:,k] <-> eval[k])
            _type_: eval_sort_index - argsort index array used to sort eigenvalues.
                    Can be used to sort via matrix[eval_sort_index]

        """

        # get eigenvalues/eigenvectors
        # eigenmode[k] is composed of eigenvector[:, k] and eigenvalue[k]
        e_val_unsorted, e_vec_unsorted = eig(self.get_transition_matrix())
        np.set_printoptions(suppress=True)  # gets rid of scientific notation

        # sort values and vectors
        eigenvalues = np.sort(e_val_unsorted)
        eval_sort_index = np.argsort(e_val_unsorted)
        eigenvalues[0] = round(eigenvalues[0])
        eigenvectors = e_vec_unsorted[:, eval_sort_index]

        # normalize eigenvector values
        eigenvectors = eigenvectors / eigenvectors[0, 0]

        if print_output:
            print("EIGENVALUES")
            print(" ", end="")
            [print(i, end=" " * 5) for i in range(self.n_spatial_locs)]
            print()
            print(eigenvalues.round(decimals=3))
            print()

            print("EIGENVECTORS")
            # eigenvector columns correspond to eigenvalues
            # (ie evec[:,k] <-> eval[k])
            print("   ", end="")
            [print(i, end=" " * 4) for i in range(self.n_spatial_locs)]
            print()
            print(eigenvectors.round(decimals=1))
            print()

        if plot_eigenmodes:
            print("EIGENMODES (e^(-eigenvalue * t))")
            ## TODO: ADD LEGEND
            self.make_eigenmode_plots(eigenvalues)

        if plot_eigenvectors:
            print("EIGENVECTORS (over space)")
            self.make_eigenvector_plots(eigenvalues, eigenvectors)

        return eigenvalues, eigenvectors, eval_sort_index

    def get_eme_init_conditions(
        self,
        print_eigenvalues_and_vectors=False,
        print_output=False,
        plot_output=False,
        plot_eigenvectors=False,
        plot_eigenmodes=False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find the initial normalized number of particles in the positive
        and negative state of eigenmode k

        params:
            eigenvectors:
                eigenvector columns correspond to eigenvalues
                (ie evec[:,k] <-> eval[k])
                eigenvector[e, eigenmode (k)]
            eval_sort_index:
                np.argsort(e_val_unsorted)
            particle_start_loc:
                location of impulse

        return:
            normalized number of particles in each eigenmode at time = 0
        """
        # starting loc given by particle_start_loc

        # get eigenvalue for starting location
        eigenvalues, eigenvectors, eval_sort_index = self.get_eigenvalues_and_vectors(
            print_output=print_eigenvalues_and_vectors,
            plot_eigenmodes=plot_eigenmodes,
            plot_eigenvectors=plot_eigenvectors,
        )

        # new index of starting node location in sorted eigenvalue/vector arrays
        start_loc_eigenvalue_i = np.where(eval_sort_index == self.particle_start_loc)[
            0
        ][
            0
        ]  # np.where returns some nested arrays, index out here

        # get eigenvector for starting location, all eigenmodes (v_k)
        start_loc_eigenvector = eigenvectors[start_loc_eigenvalue_i, :]

        # UNNORMALIZED SOLUTION
        n_per_positive_mode = 0.5 * (
            np.sqrt(self.n_particles**2 * start_loc_eigenvector**2)
            + (self.n_particles * start_loc_eigenvector)
        )

        n_per_negative_mode = 0.5 * (
            np.sqrt(self.n_particles**2 * start_loc_eigenvector**2)
            - (self.n_particles * start_loc_eigenvector)
        )

        if print_output:
            print("EIGENMODE INITIAL CONDITIONS")
            print("POSITIVE")
            print(n_per_positive_mode)
            print("NEGATIVE")
            print(n_per_negative_mode)
            print()

            # Also visualize the weights as an array
            # TODO: why do we need the [0]th
            all_init_modes = np.vstack(
                (n_per_positive_mode, n_per_negative_mode)  # , axis=0
            )

            plt.imshow(all_init_modes, interpolation="none")
            plt.yticks([0, 1], ["$+$", "$-$"])
            plt.xlabel("# Modes")
            plt.show()

        if plot_output:
            # Visualize positive and negative eigenmodes (cosines)
            # These are scaled with init coefficients

            fig, ax = plt.subplots(len(eigenvectors) + 1, 1, figsize=(4, 12))
            alpha = 1.0

            v_qp_sum = np.zeros_like(eigenvectors[:, 0])
            v_qm_sum = np.zeros_like(v_qp_sum)

            for k in range(len(eigenvectors)):
                v = eigenvectors[:, k]

                v_qp = n_per_positive_mode[k] * v
                v_qm = n_per_negative_mode[k] * v

                ax[k].plot(v_qp, "*-", c="red", alpha=alpha)
                ax[k].plot(v_qm, "*-", c="blue", alpha=alpha)
                ax[k].set_ylim([-60, 60])

                v_qp_sum += v_qp
                v_qm_sum += v_qm

                alpha *= 0.75

            ax[k + 1].plot(v_qp_sum, "*-", c="red")
            ax[k + 1].plot(v_qm_sum, "*-", c="blue")

            ax[int(k / 2)].set_ylabel("#Particles in $q_+$ or $q_-$")
            ax[k + 1].set_ylabel("Summed")
            plt.xlabel("Spatial Node")
            plt.suptitle("Initial spin states $c_k v_k$")
            plt.tight_layout()
            plt.show()

        return n_per_positive_mode, n_per_negative_mode

    def get_eigenmode_transition_probability(self, print_output=False) -> np.ndarray:
        """Get the probability of transitioning from one eigenmode to another

        Args:
            print_output (bool, optional): Whether to print outputs. Defaults to False.

        Returns:
            np.ndarray: transition probability
        """

        eigenvalues, _, _ = self.get_eigenvalues_and_vectors()
        transition_probability = (eigenvalues / 2) * self.dt

        if print_output:
            print("EIGENMODE TRANSITION PROBABILITIES")
            [print(i, end="\t") for i in range(self.n_spatial_locs)]
            print()
            [
                print("{:0.1e}".format(transition_probability[i]), end=" ")
                for i in range(self.n_spatial_locs)
            ]
            print()

        return transition_probability

    def run_simulation(
        self,
        binomial_sampling=False,
        print_eigenvalues_and_vectors=False,
        print_init_conditions=False,
        print_transition_probability=False,
        plot_eigenvectors=False,
        plot_eigenmodes=False,
        plot_init_conditions=False,
        plot_simulation=False,
        truncation_method=None,
    ) -> np.ndarray:
        """Markov simulation for eigenmode analysis to capture calcium diffusion
        over time

        params:
            n_eigenmodes:
                number of modes (k not +/- k); equal to number of locations
                (n_spatial_loc)
            init_cond:
                initial distribution of particles between positive and negative
                eigenmode states; [positive vector, negative vector]
            transition_probability:
                probability of transitioning between + and - eigenmode states
            binomial_sampling: whether to use binomial sampling or not      (default: False)
            plot: whether to plot the results or not                        (default: False)
            truncation_method:
                (experimental) attempt to address modal variance, set to:
                - None (default): no truncation method used
                - 'reflect': guarantees (q+ - q-)>=0 is always satisfied by swapping q+ and q- if (q+ - q-)<0


        return:
            n_per_eigenmode_state: normalized number of particles in each
            eigenmode (+/-) at each timepoint
            np aray shape (n_modes x n_time x n_eigenmode_states)
        """

        # positive and negative states
        n_spins = 2

        # initialize number of particles
        # n_modes x n_time x n_spins (for +/-, this is 2)
        n_per_eigenmode_state = np.zeros(
            (self.n_spatial_locs, self.n_time_pts, n_spins)
        ).astype("int")

        # assign initial conditions using number of molecules
        init_cond = self.get_eme_init_conditions(
            print_output=print_init_conditions, plot_output=plot_init_conditions
        )
        init_cond = (
            np.rint(init_cond) / self.scaling_factor
        )  # round initial conditions to nearest int

        # get transition probability
        transition_probability = self.get_eigenmode_transition_probability(
            print_output=print_transition_probability
        )

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

                # truncate if necessary
                if truncation_method == "reflect":
                    if n_spins == 2:
                        # positive - negative
                        n_per_eigenmode_init_cond = (
                            n_per_eigenmode_state[k, 0, 0]
                            - n_per_eigenmode_state[k, 0, 1]
                        )
                        n_per_eigenmode = (
                            n_per_eigenmode_state[k, i + 1, 0]
                            - n_per_eigenmode_state[k, i + 1, 1]
                        )
                        if n_per_eigenmode * n_per_eigenmode_init_cond < 0:  # crossover
                            # flip the effect of n_change above (undo, and pushback another n_change)
                            n_per_eigenmode_state[k, i + 1, 0] -= 2 * (
                                -n_change[0] + n_change[1]
                            )
                            n_per_eigenmode_state[k, i + 1, 1] -= 2 * (
                                -n_change[1] + n_change[0]
                            )
                    else:
                        print(
                            f"Truncation method {truncation_method} for n_spins={n_spins} is not implemented."
                        )

        if plot_simulation:
            n_plot_columns = 2
            n_plot_rows = math.ceil(self.n_spatial_locs / n_plot_columns)
            fig, ax = plt.subplots(n_plot_columns, n_plot_rows, figsize=(14, 10))

            m_count = 0
            for i in range(n_plot_columns):
                for j in range(n_plot_rows):
                    for k in range(n_spins):
                        if m_count < self.n_spatial_locs:
                            ax[i, j].plot(
                                list(range(self.n_time_pts)),
                                n_per_eigenmode_state[m_count, :, k],
                            )
                            ax[i, j].set_title("Eigenmode {}".format(m_count))

                    m_count += 1

            # fig.suptitle
            fig.tight_layout()
            plt.show()

        return n_per_eigenmode_state

    def convert_to_spatial_nodes(
        self,
        n_per_eigenmode_state: np.ndarray,
        print_output=False,  # print_eigenmodes_to_spatial_nodes=False,
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
        _, eigenvectors, _ = self.get_eigenvalues_and_vectors()

        # initialize node values (n_nodes x n_time_pts)
        node_vals_from_modes = np.zeros((self.n_spatial_locs, self.n_time_pts))

        # positive - negative
        n_per_eigenmode = (
            n_per_eigenmode_state[:, :, 0] - n_per_eigenmode_state[:, :, 1]
        )

        """
        A NOTE ON MATRIX MULTIPLICATION IN PYTHON
        # np.dot usage: np.dot(a, b, out=None)
        # If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
            # v: (11, 1)
            # n_per_eigenmode, m_k: (11, 1000)
            # np.dot(v, m_k): (1, 1000)
        # If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.
            # v: (11, 11)
            # n_per_eigenmode, m_k: (11, 1000)
            # np.dot(v, m_k): (11, 1000)
        """

        # for each spatial node
        for i in range(self.n_spatial_locs):
            node_vals_from_modes[i, :] = (
                np.dot(eigenvectors[i, :], n_per_eigenmode) / self.n_spatial_locs
            )

        if print_output:
            import math

            print(
                "NORMALIZED COUNT PER SPATIAL NODE (FROM EIGENMODES) (N_NODES x TIME)"
            )
            print("PRINTING SIMULATION TIME POINTS SEPARATED BY 100 TIME POINTS")
            for i in range(self.n_spatial_locs):
                print(i, end="\t")
                print(
                    (
                        node_vals_from_modes[
                            i, 0 : self.n_time_pts : math.ceil(self.n_time_pts / 14)
                        ]
                        / self.n_particles
                    ).round(decimals=1)
                )
            print()

        return self.scaling_factor * node_vals_from_modes

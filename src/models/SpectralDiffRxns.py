"""

1D diffusion equation with homogeneous Neumann conditions using spectral methods:

simulate solves the wave equation

   u_tt = c**2*u_xx + f(x,t) on

(0,L) with du/dn=0 on x=0 and x = L.
"""

import math
import numpy as np
from typing import Union
import matplotlib.pyplot as plt


class SpectralDiffRxns:
    def __init__(
        self,
        n_ca: int,  # number of calcium molecules
        n_calb: int,  # number of calcium molecules
        D_ca: float,  # calcium diffusion coefficient (um^2/usec)
        D_calb: float,  # calbindin diffusion coefficient (um^2/usec)
        kf: float,  # forward rate constant
        kr: float,  # reverse rate constant
        n_spatial_locs: int,  # define number of grid points along 1D line,
        n_time_pts: int,  # number of time points
        impulse_idx: int,  # start position of input impulse molecules
        n_eigenmodes: int,  # number of eigenmodes to use in spectral method
        dt: Union[int, float] = 1,  # time step (usec)
        line_length: Union[
            int, float
        ] = 4,  # length of line on which molecule is diffusing (um)
    ):
        self.n_ca = n_ca
        self.n_calb = n_calb
        self.D_ca = D_ca
        self.D_calb = D_calb
        self.kf = kf
        self.kr = kr
        self.n_spatial_locs = n_spatial_locs
        self.n_time_pts = n_time_pts
        self.impulse_idx = impulse_idx
        self.n_eigenmodes = n_eigenmodes
        self.dt = dt
        self.line_length = line_length
        self.ca_idx = 0
        self.calb_idx = 1
        self.ca_calb_idx = 2
        self.u = np.zeros((self.n_spatial_locs, self.n_time_pts, 3))
        self.T = np.zeros(
            (self.n_eigenmodes, self.n_time_pts, 3)
        )  # temporal component 3 species

    @property
    def time_mesh(self):
        """Return time mesh."""
        return np.linspace(0, self.n_time_pts * self.dt, self.n_time_pts)

    @property
    def spatial_mesh(self):
        """Return spatial mesh."""
        return np.linspace(0, self.line_length, self.n_spatial_locs)

    def Z_n(self, n):
        """The scaling factor associated with each eigenmodes

        NOTE: May need to add (self.n_particles / 25) scaling factor somewhere...

        Args:
            i (int): eigenmode index
        """

        if n == 0:
            return 1 / self.line_length

        else:
            return 2 / self.line_length

    def cos_n(self, n, x):
        """Gets the cosine function for the eigenmode and spatial location.

        Args:
            n (int): eigenmode index
            x (float): spatial location (um)
        """
        return np.cos(n * np.pi * x) / self.line_length

    def get_T_ca_initial_condition(self):
        """Initial condition for calcium in the temporal component across all eigenmodes."""

        ic = [
            self.n_ca * self.Z_n(n) * self.cos_n(n, self.spatial_mesh[self.impulse_idx])
            for n in range(self.n_eigenmodes)
        ]

        return ic

    def get_T_calb_initial_condition(self):
        """Initial condition for calbindin in the temporal component across all eigenmodes.

        Zeroth eigenmode is the only nonzero mode because calbindin is uniformly distributed.
        """
        ic = [0] * self.n_eigenmodes

        ic[0] = self.n_calb * self.Z_n(0)

        return ic

    def get_T_ca_calb_initial_condition(self):
        """Initial condition for ca-calbindin in the temporal component across all eigenmodes.

        None initially.
        """
        ic = [0] * self.n_eigenmodes

        return ic

    def alpha(self, i, j, n):
        """Gets the nonlinear reaction interaction term.

        Args:
            i (int): calcium eigenmode index
            j (int): calbindin eigenmode index
            n (int): ca+calbindin eigenmode index
        """

        if n == abs(i + j) or n == abs(i - j):
            alpha = (self.Z_n(i) * self.Z_n(j) * self.Z_n(n)) * (self.line_length / 4)
        else:
            alpha = 0

        return alpha

    def lambda_value(self, eigen_idx):
        """Gets the lambda value for the eigenmode.

        Args:
            eigen_idx (int): eigenmode index
        """

        return (math.pi**2) * (eigen_idx**2) / (self.line_length**2)

    def coupling_term(self, t_idx, eigen_idx):
        """Gets the coupling term for the reaction.

        Args:
            T (np.ndarray): temporal solution array
            t_idx (int): time point index
            eigen_idx (int): eigenmode index
        """
        coupling = 0
        for i in range(0, self.n_eigenmodes):
            for j in range(0, self.n_eigenmodes):
                coupling += (
                    self.alpha(i, j, eigen_idx)
                    * self.T[i, t_idx, self.ca_idx]
                    * self.T[j, t_idx, self.calb_idx]
                )

        return coupling

    def time_eqtn(self, species_idx, t_idx, eigen_idx):
        """Obtain the temporal component for a given species at a given time point and eigenmode.

        Used to update the temporal component of the solution array using the previous time point.

        Args:
            species_idx (int): index of the particle species [0, 1, 2]
            t_idx (int): time point index to use
            eigen_idx (int): eigenmode index to use

        Returns:
            _type_: temporal component of the solution array at the next time step.
        """

        sign = -1 if species_idx == self.ca_calb_idx else 1
        D = self.D_ca if species_idx == self.ca_idx else self.D_calb

        # just do for calclium for now
        time = self.T[eigen_idx, t_idx, species_idx] + self.dt * (
            (-sign * self.kf * self.coupling_term(t_idx, eigen_idx))
            + (sign * self.kr * self.T[eigen_idx, t_idx, self.ca_calb_idx])
            - (D * self.lambda_value(eigen_idx) * self.T[eigen_idx, t_idx, species_idx])
        )

        return time

    def update_eqtn(self, x_idx, t_idx, species_idx):
        u = (self.time_eqtn(species_idx, t_idx, 0) * self.Z_n(0)) + sum(
            [
                (
                    (self.n_ca / 25)  # TODO: look into scaling factor (calb?)
                    * self.Z_n(m)
                    * self.cos_n(m, self.spatial_mesh[x_idx])
                    * self.time_eqtn(species_idx, t_idx, m)
                )
                for m in range(1, self.n_eigenmodes)
            ]
        )

        return u

    def simulate(self):
        """
        Simulate calcium diffusion using finite differencing with no reactions.
        """
        # Define mesh
        x = self.spatial_mesh
        t = self.time_mesh

        # Solution arrays initialized with class

        # Define initial condition
        print("Setting initial conditions...")
        self.u[self.impulse_idx, 0, self.ca_idx] = self.n_ca
        self.u[:, 0, self.calb_idx] = int((self.n_calb) / self.n_spatial_locs)
        self.T[:, 0, self.ca_idx] = self.get_T_ca_initial_condition()
        self.T[:, 0, self.calb_idx] = self.get_T_calb_initial_condition()
        self.T[:, 0, self.ca_calb_idx] = self.get_T_ca_calb_initial_condition()

        # Solve the PDE
        print("Beginning simulation...")
        for i in range(0, len(t) - 1):
            if i % 10 == 0:
                print("Time step: ", i)
            for j in range(0, len(x)):
                for k in range(0, 3):
                    self.u[j, i + 1, k] = self.update_eqtn(j, i, k)
        print("Simulation complete!")

        return self.u

    def plot(self, t, xlim=[1, 3.5], ylim=[0, 1.1]):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        labels = ["Ca", "Calb", "Ca-Calb"]
        labels_long = ["Calcium", "Calbindin", "Bound Calcium-Calbindin"]
        total_particles = [
            self.n_ca,
            self.n_calb,
            self.n_calb,
        ]

        # plot 3 subplots, 1 for each species
        if type(t) == int:
            for i in range(3):
                print(i)
                axs[i].plot(
                    self.spatial_mesh,
                    self.u[:, t, i] / total_particles[i],
                    label=labels[i],
                )
                axs[i].set(xlabel="Distance (um)", ylabel="Normalized Calcium count")
                axs[i].set_xlim(1, 3.5)
                axs[i].set_ylim(0, 0.5)
        elif type(t) == list:
            t.reverse()
            for j in t:
                for i in range(3):
                    axs[i].plot(
                        self.spatial_mesh,
                        self.u[:, t, i] / total_particles[i],
                        label=f"t = {j}",
                    )
            for i in range(3):
                axs[i].set_title(labels_long[i])
                axs[i].set(xlabel="Distance (um)", ylabel="Normalized particle count")
                axs[i].set_xlim(1.5, 3)
                # axs[i].set_ylim(0, 1.1)

        # Set title and labels for axes
        plt.title("Spectral Calcium Diffusion with Calbindin Buffer")
        plt.xlabel("Distance (um)")
        plt.ylabel("Normalized Calcium count")
        plt.legend()

        # Set x and y limits
        # plt.xlim(xlim[0], xlim[1])
        # plt.ylim(ylim[0], ylim[1])

        plt.tight_layout()
        plt.show()

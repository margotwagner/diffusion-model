"""

1D diffusion equation with homogeneous Neumann conditions using spectral methods:

simulate solves the wave equation

   u_tt = c**2*u_xx + f(x,t) on

(0,L) with du/dn=0 on x=0 and x = L.

NOTE: scaling factor require to match the Finite Difference scheme:
    - it is a function of the number of particles initially injected into the system
    - scaling factor is found to empirically be the number of particles / 25
    - the number of time points and number of space points does not affect the scaling factor
        - unlikely to be related to concentration conversion given this
    - could maybe be related to usage of line length vs number of points in scaling factor? (TODO test more)
"""

import math
import numpy as np
from typing import Union
import matplotlib.pyplot as plt


class SpectralDiffNoRxns:
    def __init__(
        self,
        n_particles: int,  # number of molecules
        n_spatial_locs: int,  # define number of grid points along 1D line,
        n_time_pts: int,  # number of time points
        impulse_idx: int,  # start position of input impulse molecules
        n_eigenmodes: int,  # number of eigenmodes to use in spectral method
        dt: Union[int, float] = 1,  # time step (usec)
        line_length: Union[
            int, float
        ] = 4,  # length of line on which molecule is diffusing (um)
        diffusion_constant_D: float = 2.20e-4,  # Calcium diffusion coeff (um^2/usec)
    ):
        self.n_spatial_locs = n_spatial_locs
        self.dt = dt
        self.n_eigenmodes = n_eigenmodes
        self.n_time_pts = n_time_pts
        self.impulse_idx = impulse_idx
        self.line_length = line_length
        self.n_particles = n_particles
        self.diffusion_constant_D = diffusion_constant_D
        self.u = np.zeros((self.n_spatial_locs, self.n_time_pts))

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
        return np.cos((n * np.pi * x) / self.line_length)

    def temporal_eqtn(self, n, t_idx):

        return math.exp(
            -((n * math.pi / self.line_length) ** 2)
            * self.diffusion_constant_D
            * self.time_mesh[t_idx]
        )

    def spectral_eqtn(self, x_idx, t_idx):
        u = self.Z_n(0) + sum(
            [
                (
                    (self.n_particles / 25)
                    * self.Z_n(m)
                    * self.cos_n(m, self.spatial_mesh[x_idx])
                    * self.cos_n(m, self.spatial_mesh[self.impulse_idx])
                    * self.temporal_eqtn(m, t_idx)
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

        # Define initial condition
        print("Initializing solution array...")
        self.u[self.impulse_idx, 0] = self.n_particles

        # Solve the PDE
        print("Beginning simulation...")
        for i in range(0, len(t)):
            if i % 10 == 0:
                print("Time step: ", i)
            for j in range(0, len(x)):
                self.u[j, i] = self.spectral_eqtn(j, i)
        print("Simulation complete!")

        return self.u

    def plot(self, t, xlim=[1, 3.5], ylim=[0, 1.1]):
        print("Plotting...")
        fig = plt.figure()

        if type(t) == int:
            plt.plot(
                self.spatial_mesh, self.u[:, t] / self.n_particles, label=f"t = {t}"
            )
        elif type(t) == list:
            t.reverse()
            for i in t:
                plt.plot(
                    self.spatial_mesh, self.u[:, i] / self.n_particles, label=f"t = {i}"
                )

        # Set title and labels for axes
        plt.title("Spectral Calcium Diffusion with No Reactions")
        plt.xlabel("Distance (um)")
        plt.ylabel("Normalized Calcium count")
        plt.legend()

        # Set x and y limits
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])

        plt.show()

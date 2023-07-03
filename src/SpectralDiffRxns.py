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


class SpectralDiffNoRxns:
    def __init__(
        self,
        n_particles: int,  # number of molecules
        n_spatial_locs: int,  # define number of grid points along 1D line,
        n_time_pts: int,  # number of time points
        particle_start_loc: int,  # start position of input impulse molecules
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
        self.particle_start_loc = particle_start_loc
        self.line_length = line_length
        self.n_particles = n_particles
        self.diffusion_constant_D = diffusion_constant_D

    @property
    def time_mesh(self):
        """Return time mesh."""
        return np.linspace(0, self.n_time_pts * self.dt, self.n_time_pts)

    @property
    def spatial_mesh(self):
        """Return spatial mesh."""
        return np.linspace(0, self.line_length, self.n_spatial_locs)

    def simulate(self):
        """
        Simulate calcium diffusion using finite differencing with no reactions.
        """
        # Define mesh
        x = self.spatial_mesh
        t = self.time_mesh

        # Initialize solution array
        u = np.zeros((len(x), len(t)))

        # Define initial condition
        u[self.particle_start_loc, 0] = self.n_particles

        # Solve the PDE
        for i in range(0, len(t) - 1):
            for j in range(0, len(x) - 1):
                u[j, i] = (1 / self.line_length) + sum(
                    [
                        (
                            (2 / self.line_length)
                            * math.cos(m * math.pi * x[j] / self.line_length)
                            * math.cos(
                                m * math.pi * self.n_particles / self.line_length
                            )
                            * math.exp(
                                -((m * math.pi / self.line_length) ** 2)
                                * self.diffusion_constant_D
                                * t[i]
                            )
                        )
                        for m in range(1, self.n_eigenmodes)
                    ]
                )

        return 2 * u

    def plot(self, u, t):
        fig = plt.figure()

        if type(t) == int:
            plt.plot(self.spatial_mesh, u[:, t] / self.n_particles, label=f"t = {t}")
        elif type(t) == list:
            t.reverse()
            for i in t:
                plt.plot(
                    self.spatial_mesh, u[:, i] / self.n_particles, label=f"t = {i}"
                )

        # Set title and labels for axes
        plt.title("Spectral Calcium Diffusion with No Reactions")
        plt.xlabel("Distance (um)")
        plt.ylabel("Normalized Calcium count")
        plt.legend()

        # Set x and y limits
        plt.xlim(1, 3.5)
        plt.ylim(0, 0.5)

        plt.show()
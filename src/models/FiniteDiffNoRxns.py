"""

1D diffusion equation with homogeneous Neumann conditions using Finite Differencing:

simulate solves the wave equation

   u_tt = c**2*u_xx + f(x,t) on

(0,L) with du/dn=0 on x=0 and x = L.
"""

import numpy as np
from typing import Union
import matplotlib.pyplot as plt


class FiniteDiffNoRxns:
    def __init__(
        self,
        n_particles: int,  # number of molecules
        n_spatial_locs: int,  # define number of grid points along 1D line,
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
        dx = x[1] - x[0]
        dt = t[1] - t[0]
        C = self.diffusion_constant_D * (dt / (dx**2))

        # Initialize solution array
        u = np.zeros((len(x), len(t)))

        # Define initial condition
        u[self.particle_start_loc, 0] = self.n_particles

        # Solve the PDE
        for i in range(0, len(t) - 1):
            # solve internal mesh using previous time step
            for j in range(1, len(x) - 2):
                u[j, i + 1] = u[j, i] + C * (u[j + 1, i] - 2 * u[j, i] + u[j - 1, i])

            # update boundary conditions
            u[0, i + 1] = u[j, i] + C * (2 * u[j + 1, i] - 2 * u[j, i])
            u[len(x) - 1, i + 1] = u[j, i] + C * (2 * u[j - 1, i] - 2 * u[j, i])

        return u

    def plot3d(self, u):
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        # downsample to be same length
        spacing = int(len(self.time_mesh) / len(self.spatial_mesh))

        time_3d = self.time_mesh[::spacing]
        u_3d = u[:, ::spacing]

        # TODO: implement with upsampled mesh instead
        # space just redo
        # u just interpolate between points

        ax.contour3D(self.spatial_mesh, time_3d, u_3d, cmap="winter")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel("u")
        plt.show()

    def plot(self, u, t, xlim=[1, 3.5], ylim=[0, 1.1]):
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
        plt.title("Finite Difference Calcium Diffusion with No Reactions")
        plt.xlabel("Distance (um)")
        plt.ylabel("Normalized Calcium count")
        plt.legend()

        # Set x and y limits
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])

        plt.show()

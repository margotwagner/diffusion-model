"""

1D reaction-diffusion equation with homogeneous Neumann conditions using Finite Differencing:

Modeling Ca2+ + unbound calb <-> bound calb

Here, unbound calbindin molecules are initially uniformly distributed along the line, and Ca2+ molecules are injected via an impulse. Initial conditions for the state of calbindin is given by the steady state concentrations. 

# TODO: implement SNARE as well

simulate solves the wave equation

   u_tt = c**2*u_xx + f(x,t) on

(0,L) with du/dn=0 on x=0 and x = L.
"""

import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("error", RuntimeWarning)


class FiniteDiffRxns:
    def __init__(
        self,
        n_ca: int,  # number of A molecules
        n_calb: int,  # number of B molecules
        D_ca: float,  # calcium diffusion coefficient (um^2/usec)
        D_calb: float,  # calbindin diffusion coefficient (um^2/usec)
        kf: float,  # forward rate constant
        kr: float,  # reverse rate constant
        n_spatial_locs: int,  # define number of grid points along 1D line,
        n_time_pts: int,  # number of time points
        impulse_idx: int,  # start position of input impulse molecules
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
        self.dt = dt
        self.line_length = line_length

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
        C_ca = self.D_ca * (dt / (dx**2))
        C_calb = self.D_calb * (dt / (dx**2))

        # Initialize solution array
        ca = np.zeros((len(x), len(t)))
        calb = np.zeros((len(x), len(t)))
        ca_calb = np.zeros((len(x), len(t)))

        # Define initial condition
        ca[self.impulse_idx, 0] = self.n_ca
        calb[:, 0] = int((self.n_calb) / self.n_spatial_locs)

        """
        print("INITIAL CONDITIONS")
        print("CA\n", ca[self.ca_start_loc, 0])
        print("\nCALB\n", calb[:, 0])
        print("\nCA-CALB\n", ca_calb[:, 0])
        print("-" * 50, "\n")
        """

        # Solve the PDE
        for i in range(0, len(t) - 1):
            # solve internal mesh using previous time step
            for j in range(1, len(x) - 2):
                try:
                    # calcium update
                    ca[j, i + 1] = (
                        ca[j, i]
                        + C_ca * (ca[j + 1, i] - 2 * ca[j, i] + ca[j - 1, i])
                        - self.kf * ca[j, i] * calb[j, i] * dt
                        + self.kr * ca_calb[j, i] * dt
                    )

                    """
                    if i == 1:
                        print("SPATIAL LOC", j)
                        print(ca[j, i + 1])
                        print()
                        print(ca[j, i])
                        print()
                        print(C_ca * (ca[j + 1, i] - 2 * ca[j, i] + ca[j - 1, i]))
                        print()
                        print(self.kf * ca[j, i] * calb[j, i] * dt)
                        print()
                        print(self.kr * ca_calb[j, i] * dt)
                        print()
                    """

                except RuntimeWarning:
                    print("CA:", ca[j, i])
                    print("CALB:", calb[j, i])
                    quit()

                # calbindin update
                calb[j, i + 1] = (
                    calb[j, i]
                    + C_calb * (calb[j + 1, i] - 2 * calb[j, i] + calb[j - 1, i])
                    - self.kf * ca[j, i] * calb[j, i] * dt
                    + self.kr * ca_calb[j, i] * dt
                )

                # bound calcium-calbindin update
                ca_calb[j, i + 1] = (
                    ca_calb[j, i]
                    + C_calb
                    * (ca_calb[j + 1, i] - 2 * ca_calb[j, i] + ca_calb[j - 1, i])
                    + self.kf * ca[j, i] * calb[j, i] * dt
                    - self.kr * ca_calb[j, i] * dt
                )

            # update boundary conditions
            # calcium
            ca[0, i + 1] = (
                ca[j, i]
                + C_ca * (2 * ca[j + 1, i] - 2 * ca[j, i])
                - self.kf * ca[j, i] * calb[j, i] * dt
                + self.kr * ca_calb[j, i] * dt
            )

            ca[len(x) - 1, i + 1] = (
                ca[j, i]
                + C_ca * (2 * ca[j - 1, i] - 2 * ca[j, i])
                - self.kf * ca[j, i] * calb[j, i] * dt
                + self.kr * ca_calb[j, i] * dt
            )

            # calbindin
            calb[0, i + 1] = (
                calb[j, i]
                + C_calb * (2 * calb[j + 1, i] - 2 * calb[j, i])
                - self.kf * ca[j, i] * calb[j, i] * dt
                + self.kr * ca_calb[j, i] * dt
            )

            calb[len(x) - 1, i + 1] = (
                calb[j, i]
                + C_calb * (2 * calb[j - 1, i] - 2 * calb[j, i])
                - self.kf * ca[j, i] * calb[j, i] * dt
                + self.kr * ca_calb[j, i] * dt
            )

            # calcium-calbindin
            ca_calb[0, i + 1] = (
                ca_calb[j, i]
                + C_calb * (2 * ca_calb[j + 1, i] - 2 * ca_calb[j, i])
                + self.kf * ca[j, i] * calb[j, i] * dt
                - self.kr * ca_calb[j, i] * dt
            )

            ca_calb[len(x) - 1, i + 1] = (
                ca_calb[j, i]
                + C_calb * (2 * ca_calb[j - 1, i] - 2 * ca_calb[j, i])
                + self.kf * ca[j, i] * calb[j, i] * dt
                - self.kr * ca_calb[j, i] * dt
            )

            """
            print(f"TIME STEP {i}")
            print("CA\n", ca[:, i])
            print("\nCALB\n", calb[:, i])
            print("\nCA-CALB\n", ca_calb[:, i])
            print("-" * 50, "\n")
            """

        return ca, calb, ca_calb

    def plot(self, data, t):
        """_summary_

        Args:
            data (list): list containing simulation results for each species [ca, calb, ca_calb]
            t (int or list): time point(s) to plot
        """
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
                    data[i][:, t] / total_particles[i],
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
                        data[i][:, j] / total_particles[i],
                        label=f"t = {j}",
                    )
            for i in range(3):
                axs[i].set_title(labels_long[i])
                axs[i].set(xlabel="Distance (um)", ylabel="Normalized particle count")
                axs[i].set_xlim(1.5, 3)
                # axs[i].set_ylim(0, 1.1)

        # Set title and labels for axes
        fig.suptitle("Finite Difference Calcium Diffusion with Calbindin Buffer")
        plt.legend()
        plt.tight_layout()

        plt.show()

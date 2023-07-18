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
from scipy.integrate import solve_ivp


class SpectralRxnDiffusion:
    def __init__(
        self,
        n_spatial_locs: int,  # define number of grid points along 1D line,
        n_time_pts: int,  # number of time points
        impulse_idx: int,  # start position of input impulse molecules
        n_eigenmodes: int,  # number of eigenmodes to use in spectral method
    ):
        self.n_spatial_locs = n_spatial_locs
        self.n_time_pts = n_time_pts
        self.impulse_idx = impulse_idx
        self.n_eigenmodes = n_eigenmodes
        self.dt = 1  # time step (usec)
        self.line_length = 4  # length of diffusion line (um)
        self.labels = ["Ca", "Calb", "Ca-Calb"]
        self.n_species = len(self.labels)
        # amount of each species
        self.u_diff = np.zeros((self.n_spatial_locs, self.n_time_pts))
        self.u_rxndiff = np.zeros(
            (self.n_spatial_locs, self.n_time_pts, self.n_species)
        )
        # self.u = np.zeros((self.n_spatial_locs, self.n_time_pts, 3))
        # temporal component
        self.T_diff = np.zeros((self.n_eigenmodes, self.n_time_pts))
        self.T_rxndiff = np.zeros((self.n_eigenmodes, self.n_time_pts, self.n_species))

    @property
    def ca_idx(self):
        return self.labels.index("Ca")

    @property
    def calb_idx(self):
        return self.labels.index("Calb")

    @property
    def ca_calb_idx(self):
        return self.labels.index("Ca-Calb")

    @property
    def time_mesh(self):
        """Return time mesh."""
        return np.linspace(0, self.n_time_pts * self.dt, self.n_time_pts)

    @property
    def spatial_mesh(self):
        """Return spatial mesh."""
        return np.linspace(0, self.line_length, self.n_spatial_locs)

    @property
    def D_ca(self):
        """Given initial conditions from Bartol et al. 2015, return them in units compatible with the simulation scheme (um, usec, molec).

        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4595661/
        """
        D_ca = 2.2e-6  # calcium diffusion coefficient (cm^2/sec)
        D_ca = (D_ca * 1e8) / 1e6  # (um^2/usec)

        return D_ca

    @property
    def D_calb(self):
        """Given initial conditions from Bartol et al. 2015, return them in units compatible with the simulation scheme (um, usec, molec).

        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4595661/
        """
        D_calb = 0.28e-6  # calbindin diffusion coefficient (cm^2/sec)
        D_calb = (D_calb * 1e8) / 1e6  # (um^2/usec)

        return D_calb

    @property
    def kf(self):
        """forward rate constant

        Returns:
            _type_: _description_
        """
        avogadro = 6.022e23  # 1/mol

        # Volume
        x = 0.5  # um
        y = 0.5  # um
        z = 4  # um

        # Calbindin binding
        kM0M1 = 17.4e7  # 1/(M*sec)
        kH0H1 = 2.2e7  # 1/(M*sec)

        kM0M1 = ((kM0M1 * 1e15) / (avogadro * 1e6)) * (x * y)  # (1/um*sec)
        kH0H1 = ((kH0H1 * 1e15) / (avogadro * 1e6)) * (x * y)  # (1/um*sec)

        return kM0M1

    @property
    def kr(self):
        kM1M0 = 35.8  # 1/sec
        kH1H0 = 2.6  # 1/sec

        kM1M0 = kM1M0 * 1e-6  # (1/usec)
        kH1H0 = kH1H0 * 1e-6  # (1/usec)

        return kM1M0

    @property
    def n_calb(self):
        avogadro = 6.022e23  # 1/mol

        # Volume
        x = 0.5  # um
        y = 0.5  # um
        z = 4  # um

        # Initial concentrations
        c_calb = 45  # concentration of calbindin (uM)

        n_calb = (c_calb * avogadro / (1e6 * 1e15)) * (x * y * z)  # molecules

        return n_calb

    @property
    def n_ca(self):
        # Initial concentrations
        n_ca = 5275  # number of calcium particles

        return n_ca

    @property
    def dx(self):
        # Define mesh
        dx = self.spatial_mesh[1] - self.spatial_mesh[0]
        return dx

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
        return np.cos(n * np.pi * x / self.line_length)

    def get_T_ca_initial_condition(self):
        """Initial condition for calcium in the temporal component across all eigenmodes."""

        # TODO: verify if n_ca is needed here
        ic = [
            self.n_ca * self.Z_n(n) * self.cos_n(n, self.spatial_mesh[self.impulse_idx])
            for n in range(self.n_eigenmodes)
        ]

        return ic

    def get_T_calb_initial_condition(self):
        """Initial condition for calbindin in the temporal component across all eigenmodes.

        Zeroth eigenmode is the only nonzero mode because calbindin is uniformly distributed.
        """
        # TODO: verify if n_calb is needed here
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
            n (int): self eigenmode index
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

    def coupling_term(self, T_eqtns, eigen_idx):
        """Gets the coupling term for the reaction.

        Args:
            T (np.ndarray): temporal solution array
            t_idx (int): time point index
            eigen_idx (int): eigenmode index
        """
        """
        coupling = 0
        for i in range(0, self.n_eigenmodes):
            for j in range(0, self.n_eigenmodes):
                coupling += (
                    self.alpha(i, j, eigen_idx)
                    * self.T[i, t_idx, self.ca_idx]
                    * self.T[j, t_idx, self.calb_idx]
                )
        """

        coupling = 0
        for ca_eigen_idx in range(0, self.n_eigenmodes):
            for calb_eigen_idx in range(0, self.n_eigenmodes):
                coupling += (
                    self.alpha(ca_eigen_idx, calb_eigen_idx, eigen_idx)
                    * T_eqtns[ca_eigen_idx]
                    * T_eqtns[self.n_eigenmodes + calb_eigen_idx]
                )

        return coupling

    def dTdt(self, t, T_eqtns, T_idx, species_idx, eigen_idx):
        """Time derivative of the temporal component of the solution array for
        an individual species and eigenmode combinations.
        """

        sign = -1 if species_idx == self.ca_calb_idx else 1
        D = self.D_ca if species_idx == self.ca_idx else self.D_calb

        dTdt = (
            (-sign * self.kf * self.coupling_term(T_eqtns, eigen_idx))
            + (
                sign
                * self.kr
                * T_eqtns[(self.ca_calb_idx * self.n_eigenmodes) + eigen_idx]
            )
            - (D * self.lambda_value(eigen_idx) * T_eqtns[T_idx])
        )

        return dTdt

    def dTdt_system(self, t, T_eqtns):
        """Right-hand side of the system of ODEs.

        The time derivative of the states T at time t. An N-D vector-valued function f(t,y).

        Args:
            t (_type_): _description_
            T_eqtns (nd.array): N-D vector-valued state function y(t)

        Returns:
            nd.array: differential equations
        """
        dTdt_eqtns = []

        T_idx = 0
        for species_idx in range(self.n_species):
            for eigen_idx in range(self.n_eigenmodes):
                dTdt_eqtns.append(self.dTdt(t, T_eqtns, T_idx, species_idx, eigen_idx))
                T_idx += 1

        return dTdt_eqtns

    def solve_dTdt(self):

        # set ICs
        self.T[:, 0, self.ca_idx] = self.get_T_ca_initial_condition()
        self.T[:, 0, self.calb_idx] = self.get_T_calb_initial_condition()
        self.T[:, 0, self.ca_calb_idx] = self.get_T_ca_calb_initial_condition()
        print("Initializing dTdt solver...")
        T0 = []
        for species_idx in range(self.n_species):
            for eigen_idx in range(self.n_eigenmodes):
                T0.append(self.T[eigen_idx, 0, species_idx])

        # solve dTdt
        print("Solving dTdt...")
        sol = solve_ivp(
            self.dTdt_system,
            [0, self.n_time_pts],
            T0,
            t_eval=self.time_mesh,
        )
        for species_idx in range(self.n_species):
            self.T[:, :, species_idx] = sol.y[
                species_idx * self.n_eigenmodes : (species_idx + 1) * self.n_eigenmodes,
                :,
            ]

        return sol

    def plot_T(self):
        print("Plotting T...")
        fig, axs = plt.subplots(2, 3, figsize=(15, 5))
        labels = ["Ca", "Calb", "Ca-Calb"]
        labels_long = ["Calcium", "Calbindin", "Bound Calcium-Calbindin"]

        # plot with time on the x-axis
        for i in range(self.n_species):
            for m in range(self.n_eigenmodes):
                if m % 1 == 0:
                    axs[0, i].plot(
                        self.time_mesh,
                        self.T[m, :, i],
                        label=f"m = {m}",
                    )

        # plot with eigenmodes on the x-axis
        for i in range(self.n_species):
            for t in range(self.n_time_pts):
                if m % 1 == 0:
                    axs[1, i].plot(
                        [*range(0, self.n_eigenmodes)],
                        self.T[:, t, i],
                        label=f"m = {t}",
                    )

        for i in range(3):
            axs[0, i].set_title(labels_long[i])
            axs[0, i].set(xlabel="time (usec)", ylabel="T(t)")
            axs[1, i].set_title(labels_long[i])
            axs[1, i].set(xlabel="eigenmodes", ylabel="T(t)")

        plt.tight_layout()
        # plt.legend()
        plt.show()

    def solve_u(self):
        """
        Solve for u(x,t) using the method of eigenfunction expansion.
        """
        # Define mesh
        x = self.spatial_mesh
        t = self.time_mesh

        # Define initial condition
        print("Setting initial conditions...")
        self.u[self.impulse_idx, 0, self.ca_idx] = self.n_ca
        self.u[:, 0, self.calb_idx] = int((self.n_calb) / self.n_spatial_locs)

        # Solve the PDE
        # TODO: make u a function
        print("Beginning simulation...")
        for time_idx in range(len(t)):
            if time_idx % 10 == 0:
                print(f"t = {time_idx}")
            for space_idx in range(len(x)):
                for species_idx in range(self.n_species):
                    self.u[space_idx, time_idx, species_idx] = self.T[
                        0, time_idx, species_idx
                    ] * self.Z_n(0) + sum(
                        [
                            (
                                self.Z_n(eigen_idx)
                                * self.cos_n(eigen_idx, self.spatial_mesh[space_idx])
                                * self.T[eigen_idx, time_idx, species_idx]
                            )
                            for eigen_idx in range(1, self.n_eigenmodes)
                        ]
                    )
        print("Simulation complete!")

        return self.u

    def plot_u(self):
        fig, axs = plt.subplots(2, 3, figsize=(15, 5))
        labels = ["Ca", "Calb", "Ca-Calb"]
        labels_long = ["Calcium", "Calbindin", "Bound Calcium-Calbindin"]

        # plot with time on the x-axis
        for i in range(self.n_species):
            for j in range(self.n_spatial_locs):
                if j % 1 == 0:
                    axs[0, i].plot(
                        self.time_mesh,
                        self.u[j, :, i],
                        label=f"x = {j}",
                    )

        # plot with space on the x-axis
        for i in range(self.n_species):
            for t in range(self.n_time_pts):
                if t % 1 == 0:
                    axs[1, i].plot(
                        self.spatial_mesh,
                        self.u[:, t, i],
                        label=f"m = {t}",
                    )

        for i in range(3):
            axs[0, i].set_title(labels_long[i])
            axs[0, i].set(xlabel="time (usec)", ylabel="u(t)")
            axs[1, i].set_title(labels_long[i])
            axs[1, i].set(xlabel="space (um)", ylabel="u(t)")

        plt.tight_layout()
        # plt.legend()
        plt.show()

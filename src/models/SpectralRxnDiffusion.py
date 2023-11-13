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
import matplotlib.pyplot as plt
import os
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
        # temporal component
        self.T = np.zeros((self.n_eigenmodes, self.n_time_pts, self.n_species))

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

        NOTE: May need to add (self.n_ca / 25) scaling factor somewhere...

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

    def diffusion_temporal_eqtn(self, n, t_idx):

        return math.exp(
            -((n * math.pi / self.line_length) ** 2) * self.D_ca * self.time_mesh[t_idx]
        )

    def diffusion_spectral_eqtn(self, x_idx, t_idx):
        u = self.Z_n(0) + sum(
            [
                (
                    (self.n_ca / (self.n_spatial_locs / self.line_length))
                    * self.Z_n(m)
                    * self.cos_n(m, self.spatial_mesh[x_idx])
                    * self.cos_n(m, self.spatial_mesh[self.impulse_idx])
                    * self.diffusion_temporal_eqtn(m, t_idx)
                )
                for m in range(1, self.n_eigenmodes)
            ]
        )

        return u

    def simulate_diffusion(self):
        """
        Simulate calcium diffusion using finite differencing with no reactions.
        """
        # Define initial condition
        print("Initializing solution array...")
        self.u_diff[self.impulse_idx, 0] = self.n_ca

        # Solve the PDE
        print("Beginning simulation...")
        for time_idx in range(0, self.n_time_pts):
            if time_idx % 10 == 0:
                print("Time step: ", time_idx)
            for space_idx in range(0, self.n_spatial_locs):
                self.u_diff[space_idx, time_idx] = self.diffusion_spectral_eqtn(
                    space_idx, time_idx
                )
        print("Simulation complete!")

        return self.u_diff

    def plot_diffusion(self, t):
        # TODO: move to separate file
        print("Plotting...")
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        # plot with space on the x-axis
        for i in t:
            axs[0].plot(
                self.spatial_mesh, self.u_diff[:, i] / self.n_ca, label=f"t = {i}"
            )
        axs[0].set_xlabel("Distance (um)")
        axs[0].set_ylabel("Normalized Calcium count")
        axs[0].set_title("Calcium vs Distance")
        axs[0].set_xlim([1.5, 3])
        axs[0].legend(title="time steps")
        axs[0].annotate(
            "A", xy=(-0.11, 1.05), xycoords="axes fraction", fontsize=16, weight="bold"
        )

        # plot with time on the x-axis
        x_idx = [self.impulse_idx + i for i in range(0, 10)]
        x_labels = [*range(0, 10)]
        for i in range(len(x_idx)):
            axs[1].plot(
                self.time_mesh,
                self.u_diff[x_idx[i], :] / self.n_ca,
                label=f"$\Delta$x = {x_labels[i]}",
            )
        axs[1].set_xlabel("Time (usec)")
        axs[1].set_title("Calcium vs Time")
        axs[1].legend(title="steps from impulse")
        axs[1].annotate(
            "B", xy=(0, 1.05), xycoords="axes fraction", fontsize=16, weight="bold"
        )

        fig.suptitle("Spectral Calcium Diffusion with No Reactions", fontsize=18)

        plt.tight_layout()
        plt.savefig("../figures/spectral-diff-norxns.png")
        plt.show()

    def get_T_ca_initial_condition(self):
        """Initial condition for calcium in the temporal component across all eigenmodes."""

        ic = [
            self.n_ca * self.Z_n(n) * self.cos_n(n, self.spatial_mesh[self.impulse_idx])
            for n in range(self.n_eigenmodes)
        ]

        """
        ic = [
            (self.n_ca * 2 / (self.n_spatial_locs / self.line_length))
            * self.Z_n(n)
            * self.cos_n(n, self.spatial_mesh[self.impulse_idx])
            for n in range(self.n_eigenmodes)
        ]
        """

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

        # coupling = 0
        # for ca_eigen_idx in range(0, self.n_eigenmodes):
        #     for calb_eigen_idx in range(0, self.n_eigenmodes):
        #         coupling += (
        #             self.alpha(ca_eigen_idx, calb_eigen_idx, eigen_idx)
        #             * T_eqtns[ca_eigen_idx]
        #             * T_eqtns[self.n_eigenmodes + calb_eigen_idx]
        #         )

        # return coupling
        return 0

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

    def solve_dTdt(self, save_dir):

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
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        np.save(save_dir, self.T)

        return sol

    def simulate_rxn_diffusion(self, save_dir):
        """
        Solve for u(x,t) using the method of eigenfunction expansion.
        """
        # Define initial condition
        print("Setting initial conditions...")
        self.u_rxndiff[self.impulse_idx, 0, self.ca_idx] = self.n_ca
        self.u_rxndiff[:, 0, self.calb_idx] = int((self.n_calb) / self.n_spatial_locs)

        # Solve the PDE
        print("Beginning simulation...")
        for time_idx in range(self.n_time_pts):
            if time_idx % 10 == 0:
                print(f"t = {time_idx}")
            for space_idx in range(self.n_spatial_locs):
                for species_idx in range(self.n_species):
                    self.u_rxndiff[space_idx, time_idx, species_idx] = (
                        self.T[0, time_idx, species_idx] * self.Z_n(0)
                    ) + sum(
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

        np.save(save_dir, self.u_rxndiff)

        return self.u_rxndiff

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
import matplotlib.pyplot as plt


class FiniteDiffRxnDiffusion:
    def __init__(
        self,
        n_spatial_locs: int,  # define number of grid points along 1D line,
        n_time_pts: int,  # number of time points
        impulse_idx: int,  # start position of input impulse molecules
    ):
        self.n_spatial_locs = n_spatial_locs
        self.n_time_pts = n_time_pts
        self.impulse_idx = impulse_idx
        self.dt = 1  # time step (usec)
        self.line_length = 4  # length of diffusion line (um)
        self.labels = ["Ca", "Calb", "Ca-Calb"]
        self.n_species = len(self.labels)
        self.u_diff = np.zeros((self.n_spatial_locs, self.n_time_pts))
        self.u_rxndiff = np.zeros(
            (self.n_spatial_locs, self.n_time_pts, self.n_species)
        )

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

    @property
    def ca_mesh_const(self):
        return self.D_ca * (self.dt / (self.dx**2))

    @property
    def calb_mesh_const(self):
        return self.D_calb * (self.dt / (self.dx**2))

    def is_stable(self):
        """von Neumann stability analysis for the diffusion equation."""

        # Calculate the stability constant
        threshold = (self.dx**2) / (2 * self.D_ca)

        if self.dt <= threshold:
            print(f"Stability condition satisfied: {self.dt} <= {threshold}")
        else:
            print(f"Stability condition NOT satisfied: {self.dt} > {threshold}")
            print("Try decreasing the time step or increasing the space step.")
            print("Exiting...")

        return self.dt <= threshold

    def ca_norxn_update(self, space_idx, time_idx):
        if space_idx == 0:
            u = self.u_diff[space_idx, time_idx] + self.ca_mesh_const * (
                2 * self.u_diff[space_idx + 1, time_idx]
                - 2 * self.u_diff[space_idx, time_idx]
            )
        elif space_idx == self.n_spatial_locs - 1:
            u = self.u_diff[space_idx, time_idx] + self.ca_mesh_const * (
                2 * self.u_diff[space_idx - 1, time_idx]
                - 2 * self.u_diff[space_idx, time_idx]
            )
        else:
            u = self.u_diff[space_idx, time_idx] + self.ca_mesh_const * (
                self.u_diff[space_idx + 1, time_idx]
                - 2 * self.u_diff[space_idx, time_idx]
                + self.u_diff[space_idx - 1, time_idx]
            )
        return u

    def simulate_diffusion(self):
        """
        Simulate calcium diffusion using finite differencing with no reactions.
        """

        # Check stability
        stable = self.is_stable()

        if not stable:
            quit()

        # Define initial condition
        print("Initializing solution array...")
        self.u_diff[self.impulse_idx, 0] = self.n_ca

        # Solve the PDE
        print("Beginning simulation...")
        for time_idx in range(0, self.n_time_pts - 1):
            if time_idx % 10 == 0:
                print("Time step: ", time_idx)

            # solve internal mesh using previous time step
            for space_idx in range(1, self.n_spatial_locs - 2):
                self.u_diff[space_idx, time_idx + 1] = self.ca_norxn_update(
                    space_idx, time_idx
                )

            # update boundary conditions
            self.u_diff[0, time_idx + 1] = self.ca_norxn_update(0, time_idx)
            self.u_diff[self.n_spatial_locs - 1, time_idx + 1] = self.ca_norxn_update(
                self.n_spatial_locs - 1, time_idx
            )
        print("Simulation complete!")

        return self.u_diff

    def ca_rxn_update(self, space_idx, time_idx):
        # Boundary updates
        if space_idx == 0:
            u = (
                self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                + self.ca_mesh_const
                * (
                    2 * self.u_rxndiff[space_idx + 1, time_idx, self.ca_idx]
                    - 2 * self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                )
                - self.kf
                * self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                * self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                * self.dt
                + self.kr
                * self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                * self.dt
            )

        elif space_idx == self.n_spatial_locs - 1:
            u = (
                self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                + self.ca_mesh_const
                * (
                    2 * self.u_rxndiff[space_idx - 1, time_idx, self.ca_idx]
                    - 2 * self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                )
                - self.kf
                * self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                * self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                * self.dt
                + self.kr
                * self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                * self.dt
            )

        else:
            u = (
                self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                + self.ca_mesh_const
                * (
                    self.u_rxndiff[space_idx + 1, time_idx, self.ca_idx]
                    - 2 * self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                    + self.u_rxndiff[space_idx - 1, time_idx, self.ca_idx]
                )
                - self.kf
                * self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                * self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                * self.dt
                + self.kr
                * self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                * self.dt
            )
        return u

    def calb_rxn_update(self, space_idx, time_idx):
        if space_idx == 0:
            u = (
                self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                + self.calb_mesh_const
                * (
                    2 * self.u_rxndiff[space_idx + 1, time_idx, self.calb_idx]
                    - 2 * self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                )
                - self.kf
                * self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                * self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                * self.dt
                + self.kr
                * self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                * self.dt
            )

        elif space_idx == self.n_spatial_locs - 1:
            u = (
                self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                + self.calb_mesh_const
                * (
                    2 * self.u_rxndiff[space_idx - 1, time_idx, self.calb_idx]
                    - 2 * self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                )
                - self.kf
                * self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                * self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                * self.dt
                + self.kr
                * self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                * self.dt
            )

        else:
            u = (
                self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                + self.calb_mesh_const
                * (
                    self.u_rxndiff[space_idx + 1, time_idx, self.calb_idx]
                    - 2 * self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                    + self.u_rxndiff[space_idx - 1, time_idx, self.calb_idx]
                )
                - self.kf
                * self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                * self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                * self.dt
                + self.kr
                * self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                * self.dt
            )
        return u

    def ca_calb_rxn_update(self, space_idx, time_idx):
        if space_idx == 0:
            u = (
                self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                + self.calb_mesh_const
                * (
                    2 * self.u_rxndiff[space_idx + 1, time_idx, self.ca_calb_idx]
                    - 2 * self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                )
                + self.kf
                * self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                * self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                * self.dt
                - self.kr
                * self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                * self.dt
            )

        elif space_idx == self.n_spatial_locs - 1:
            u = (
                self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                + self.calb_mesh_const
                * (
                    2 * self.u_rxndiff[space_idx - 1, time_idx, self.ca_calb_idx]
                    - 2 * self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                )
                + self.kf
                * self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                * self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                * self.dt
                - self.kr
                * self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                * self.dt
            )

        else:
            u = (
                self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                + self.calb_mesh_const
                * (
                    self.u_rxndiff[space_idx + 1, time_idx, self.ca_calb_idx]
                    - 2 * self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                    + self.u_rxndiff[space_idx - 1, time_idx, self.ca_calb_idx]
                )
                + self.kf
                * self.u_rxndiff[space_idx, time_idx, self.ca_idx]
                * self.u_rxndiff[space_idx, time_idx, self.calb_idx]
                * self.dt
                - self.kr
                * self.u_rxndiff[space_idx, time_idx, self.ca_calb_idx]
                * self.dt
            )
        return u

    def simulate_rxn_diffusion(self):
        """
        Simulate calcium diffusion using finite differencing with no reactions.
        """

        # Define initial condition
        print("Initializing solution array...")
        self.u_rxndiff[self.impulse_idx, 0, self.ca_idx] = self.n_ca
        self.u_rxndiff[:, 0, self.calb_idx] = int((self.n_calb) / self.n_spatial_locs)

        # Solve the PDE
        print("Beginning simulation...")
        for time_idx in range(0, self.n_time_pts - 1):
            if time_idx % 10 == 0:
                print("Time step: ", time_idx)
            # solve internal mesh using previous time step
            for space_idx in range(1, self.n_spatial_locs - 2):
                # calcium update
                self.u_rxndiff[
                    space_idx, time_idx + 1, self.ca_idx
                ] = self.ca_rxn_update(space_idx, time_idx)

                # calbindin update
                self.u_rxndiff[
                    space_idx, time_idx + 1, self.calb_idx
                ] = self.calb_rxn_update(space_idx, time_idx)

                # bound calcium-calbindin update
                self.u_rxndiff[
                    space_idx, time_idx + 1, self.ca_calb_idx
                ] = self.ca_calb_rxn_update(space_idx, time_idx)

            # update boundary conditions
            # calcium
            self.u_rxndiff[0, time_idx + 1, self.ca_idx] = self.ca_rxn_update(
                0, time_idx
            )

            self.u_rxndiff[
                self.n_spatial_locs - 1, time_idx + 1, self.ca_idx
            ] = self.ca_rxn_update(self.n_spatial_locs - 1, time_idx)

            # calbindin
            self.u_rxndiff[0, time_idx + 1, self.calb_idx] = self.calb_rxn_update(
                0, time_idx
            )

            self.u_rxndiff[
                self.n_spatial_locs - 1, time_idx + 1, self.calb_idx
            ] = self.calb_rxn_update(self.n_spatial_locs - 1, time_idx)

            # calcium-calbindin
            self.u_rxndiff[0, time_idx + 1, self.ca_calb_idx] = self.ca_calb_rxn_update(
                0, time_idx
            )

            self.u_rxndiff[
                self.n_spatial_locs - 1, time_idx + 1, self.ca_calb_idx
            ] = self.ca_calb_rxn_update(self.n_spatial_locs - 1, time_idx)

        print("Simulation complete!")

        return self.u_rxndiff

    def plot_diffusion(self, t):
        print("Plotting...")
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

        # plot with space on the x-axis
        for i in t:
            axs[0].plot(
                self.spatial_mesh,
                self.u_diff[:, i] / self.n_ca,
                label=f"t = {i}",
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

        fig.suptitle(
            "Finite Difference Calcium Diffusion with No Reactions", fontsize=18
        )

        plt.tight_layout()
        plt.savefig("../figures/finite-diff-norxns.png")
        plt.show()

    def plot_rxn_diffusion(self, t, orientation="vertical"):
        """_summary_

        Args:
            data (list): list containing simulation results for each species [ca, calb, ca_calb]
            t (int or list): time point(s) to plot
        """
        if orientation == "horizontal":
            fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        else:
            fig, axs = plt.subplots(3, 2, figsize=(10, 10))

        labels_long = ["Calcium", "Calbindin", "Bound Calcium-Calbindin"]
        total_particles = [
            self.n_ca,
            self.n_calb,
            self.n_calb,
        ]

        # plot 3 subplots, 1 for each species
        # plot with space on the x-axis
        for time_idx in t:
            for i in range(self.n_species):
                if orientation == "horizontal":
                    term_1, term_2 = 0, i
                else:
                    term_1, term_2 = i, 0
                axs[term_1, term_2].plot(
                    self.spatial_mesh,
                    self.u_rxndiff[:, time_idx, i] / total_particles[i],
                    label=f"t = {time_idx}",
                )

        # plot with time on the x-axis
        delta_xs = [*range(0, 10)]
        x_idx = [self.impulse_idx + i for i in delta_xs]
        for x_i in range(len(x_idx)):
            for species in range(self.n_species):
                if orientation == "horizontal":
                    term_1, term_2 = 1, species
                else:
                    term_1, term_2 = species, 1

                axs[term_1, term_2].plot(
                    self.time_mesh,
                    self.u_rxndiff[x_idx[x_i], :, species] / total_particles[species],
                    label=f"$\Delta$x = {delta_xs[x_i]}",
                )

        # Add labels
        xlabs = ["distance (um)", "time (usec)"]
        for i in range(2):
            for j in range(self.n_species):
                if orientation == "horizontal":
                    term_1, term_2 = i, j
                else:
                    term_1, term_2 = j, i
                if i == 0:
                    axs[term_1, term_2].set_xlim(1.5, 3)
                axs[term_1, term_2].set(
                    xlabel=xlabs[i], ylabel="Normalized particle count"
                )
                axs[term_1, term_2].set_title(labels_long[j])

        # Set limits
        if orientation == "horizontal":
            term_1, term_2 = 0, 1
        else:
            term_1, term_2 = 1, 0

        axs[term_1, term_2].set_ylim([6.6225e-3, 6.643e-3])
        axs[term_1, term_2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        # Add legends
        if orientation == "horizontal":
            axs[0, 2].legend(
                title="time step", loc="upper right", bbox_to_anchor=(1.4, 1)
            )
            axs[1, 2].legend(
                title="steps from impulse", loc="upper right", bbox_to_anchor=(1.5, 1)
            )
        else:
            axs[0, 0].legend(title="time step", loc="upper left")
            axs[0, 1].legend(title="steps from impulse", loc="upper right", ncol=2)

        # Add letters
        letters = ["A", "B", "C", "D", "E", "F"]
        ax = axs.flatten()

        for i in range(6):
            ax[i].annotate(
                letters[i],
                xy=(-0.1, 1.05),
                xycoords="axes fraction",
                fontsize=16,
                weight="bold",
            )

        # Set title and save
        fig.suptitle(
            "Finite Difference Calcium Diffusion with Calbindin Buffer", fontsize=18
        )
        plt.tight_layout()
        plt.savefig(f"../figures/finite-diff-rxns-{orientation}.png", dpi=500)
        plt.show()

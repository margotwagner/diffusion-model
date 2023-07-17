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
    print("Beginning simulation...")
    for i in range(0, len(t) - 1):
        for j in range(0, len(x)):
            for k in range(self.n_species):
                self.u[j, i + 1, k] = self.T[0, i, k] * self.Z_n(0) + sum(
                    [
                        (
                            self.Z_n(m)
                            * self.cos_n(m, self.spatial_mesh[j])
                            * self.T[m, i, k]
                        )
                        for m in range(1, self.n_eigenmodes)
                    ]
                )
    print("Simulation complete!")

    return self.u


def dTdt_system_ugly(self, t, T):

    T_ca_0, T_ca_1, T_calb_0, T_calb_1, T_ca_calb_0, T_ca_calb_1 = T

    dTdt = [
        -self.kf
        * (
            self.alpha(0, 0, 0) * T_ca_0 * T_calb_0
            + self.alpha(1, 1, 0) * T_ca_1 * T_calb_1
            + self.alpha(0, 1, 0) * T_ca_0 * T_calb_1
            + self.alpha(1, 0, 0) * T_ca_1 * T_calb_0
        )
        + self.kr * T_ca_calb_0
        + self.dTdt_noreact(t, T_ca_0, self.ca_idx, 0),
        -self.kf
        * (
            self.alpha(0, 0, 1) * T_ca_0 * T_calb_0
            + self.alpha(1, 1, 1) * T_ca_1 * T_calb_1
            + self.alpha(0, 1, 1) * T_ca_0 * T_calb_1
            + self.alpha(1, 0, 1) * T_ca_1 * T_calb_0
        )
        + self.kr * T_ca_calb_1
        + self.dTdt_noreact(t, T_ca_1, self.ca_idx, 1),
        -self.kf
        * (
            self.alpha(0, 0, 0) * T_ca_0 * T_calb_0
            + self.alpha(1, 1, 0) * T_ca_1 * T_calb_1
            + self.alpha(0, 1, 0) * T_ca_0 * T_calb_1
            + self.alpha(1, 0, 0) * T_ca_1 * T_calb_0
        )
        + self.kr * T_ca_calb_0
        + self.dTdt_noreact(t, T_calb_0, self.ca_idx, 0),
        -self.kf
        * (
            self.alpha(0, 0, 1) * T_ca_0 * T_calb_0
            + self.alpha(1, 1, 1) * T_ca_1 * T_calb_1
            + self.alpha(0, 1, 1) * T_ca_0 * T_calb_1
            + self.alpha(1, 0, 1) * T_ca_1 * T_calb_0
        )
        + self.kr * T_ca_calb_1
        + self.dTdt_noreact(t, T_calb_1, self.ca_idx, 1),
        self.kf
        * (
            self.alpha(0, 0, 0) * T_ca_0 * T_calb_0
            + self.alpha(1, 1, 0) * T_ca_1 * T_calb_1
            + self.alpha(0, 1, 0) * T_ca_0 * T_calb_1
            + self.alpha(1, 0, 0) * T_ca_1 * T_calb_0
        )
        - self.kr * T_ca_calb_0
        + self.dTdt_noreact(t, T_ca_calb_0, self.ca_idx, 0),
        self.kf
        * (
            self.alpha(0, 0, 1) * T_ca_0 * T_calb_0
            + self.alpha(1, 1, 1) * T_ca_1 * T_calb_1
            + self.alpha(0, 1, 1) * T_ca_0 * T_calb_1
            + self.alpha(1, 0, 1) * T_ca_1 * T_calb_0
        )
        - self.kr * T_ca_calb_1
        + self.dTdt_noreact(t, T_ca_calb_1, self.ca_idx, 1),
    ]

    return dTdt


def solve_dTdt_ugly(self):
    # set ICs
    self.T[:, 0, self.ca_idx] = self.get_T_ca_initial_condition()
    self.T[:, 0, self.calb_idx] = self.get_T_calb_initial_condition()
    self.T[:, 0, self.ca_calb_idx] = self.get_T_ca_calb_initial_condition()

    T0 = [
        self.T[0, 0, self.ca_idx],
        self.T[1, 0, self.ca_idx],
        self.T[0, 0, self.calb_idx],
        self.T[1, 0, self.calb_idx],
        self.T[0, 0, self.ca_calb_idx],
        self.T[1, 0, self.ca_calb_idx],
    ]

    sol = solve_ivp(
        self.dTdt_system_ugly,
        [0, self.n_time_pts],
        T0,
        t_eval=self.time_mesh,
    )
    self.T[:, :, :] = np.reshape(sol.y, (self.n_eigenmodes, self.n_time_pts, 3))

    return sol

import FiniteDiffNoRxns
import SpectralDiffNoRxns
import FiniteDiffRxns


def get_D_ca():
    """Given initial conditions from Bartol et al. 2015, return them in units compatible with the simulation scheme (um, usec, molec).

    Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4595661/
    """
    D_ca = 2.2e-6  # calcium diffusion coefficient (cm^2/sec)
    D_ca = (D_ca * 1e8) / 1e6  # (um^2/usec)

    return D_ca


def get_D_ca():
    """Given initial conditions from Bartol et al. 2015, return them in units compatible with the simulation scheme (um, usec, molec).

    Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4595661/
    """
    D_calb = 0.28e-6  # calbindin diffusion coefficient (cm^2/sec)
    D_calb = (D_calb * 1e8) / 1e6  # (um^2/usec)

    return D_calb


def get_ca_init_loc():
    # VDCC
    vdcc_loc = 2.35  # um

    return vdcc_loc


def get_kf():
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


def get_kr():
    kM1M0 = 35.8  # 1/sec
    kH1H0 = 2.6  # 1/sec

    kM1M0 = kM1M0 * 1e-6  # (1/usec)
    kH1H0 = kH1H0 * 1e-6  # (1/usec)

    return kM1M0


def get_n_ca():
    # Initial concentrations
    n_ca = 5275  # number of calcium particles

    return n_ca


def get_n_calb():
    avogadro = 6.022e23  # 1/mol

    # Volume
    x = 0.5  # um
    y = 0.5  # um
    z = 4  # um

    # Initial concentrations
    c_calb = 45  # concentration of calbindin (uM)

    n_calb = (c_calb * avogadro / (1e6 * 1e15)) * (x * y * z)  # molecules

    return n_calb


def get_ca_init_idx(n_space_pts):
    vdcc_loc = get_ca_init_loc()

    z = 4  # um
    dx = z / n_space_pts

    return int(vdcc_loc / dx)


def main():
    n_ca = get_n_ca()  # number of calcium particles
    n_calb = get_n_calb()  # number of calbindin particles
    kf = get_kf()  # forward rate constant (molec/um/usec)
    kr = get_kr()  # reverse rate constant (1/usec)
    n_time_pts = 100  # number of time points
    n_space_pts = 100  # number of spatial points
    ca_init_idx = get_ca_init_idx(n_space_pts)

    # Finite Difference (no reactions)
    fd = FiniteDiffNoRxns.FiniteDiffNoRxns(
        n_particles=n_ca,
        n_spatial_locs=n_space_pts,
        n_time_pts=n_time_pts,
        particle_start_loc=ca_init_idx,
    )

    # Finite Difference (calbindin reactions)
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4595661/
    fd_rxn = FiniteDiffRxns.FiniteDiffRxns(
        n_ca_particles=n_ca,
        n_calb_particles=n_calb,
        n_spatial_locs=n_space_pts,
        n_time_pts=n_time_pts,
        ca_start_loc=ca_init_idx,
        kf=kf,
        kr=kr,
    )

    # Spectral Diffusion (no reactions)
    sd = SpectralDiffNoRxns.SpectralDiffNoRxns(
        n_particles=50,
        n_spatial_locs=n_space_pts,
        n_time_pts=1000,
        particle_start_loc=ca_init_idx,
        n_eigenmodes=n_space_pts,
    )

    # fd_u = fd.simulate()
    sd_u = sd.simulate()
    # ca, calb, ca_calb = fd_rxn.simulate()

    # Finite Differencing No Reactions
    # fd.plot(fd_u, [0, 1, 5, 20, 40, 50, 99])
    # fd.plot(fd_u, [5, 20, 40, 50, 99])

    # Spectral Diffusion No Reactions
    #sd.plot(sd_u, [0, 1, 5, 20, 40, 50, 99])
    sd.plot(sd_u, [5, 20, 40, 50, 99])

    # Finite Differencing with Reactions
    # fd_rxn.plot([ca, calb, ca_calb], [0, 1, 2, 3, 4, 5, 10, 20, 40, 50, 99])


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt


def percent_error(true_load_dir, exp_load_dir, save_dir):
    # Initialize simulation data
    u_true = np.load(true_load_dir)
    u_exp = np.load(exp_load_dir)

    n_ca_exp = np.max(u_exp[:, 0, 0])
    n_calb_exp = np.sum(u_exp[:, 0, 1])
    total_particles_exp = [
        n_ca_exp,
        n_calb_exp,
        n_calb_exp,
    ]
    scaling_factor_exp = [
        1,
        1.003387,
        4,
    ]
    spatial_mesh = np.linspace(0, 4, u_exp.shape[0])
    time_mesh = [*range(0, u_exp.shape[1])]

    # scale results if necessary
    for species in range(len(total_particles_exp)):
        u_exp[:, :, species] = (
            u_exp[:, :, species]
            / total_particles_exp[species]
            / scaling_factor_exp[species]
        )

        # TODO: scale finite diff results
        # u_true[:, :, species] = u_true[:, :, species] / total_particles_exp[species] / scaling_factor_exp[species]

    # get percent_error for each space/time step
    error = np.abs(u_true - u_exp) / u_true

    # get total percent_error for each species
    total_error_by_species = np.sum(error, axis=(0, 1))
    total_error_over_time = np.sum(error, axis=0)
    total_error_over_space = np.sum(error, axis=1)
    total_error = np.sum(total_error_by_species)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # plot percent_error vs time step
    # plot with time on the x-axis
    delta_xs = [*range(0, 10)]
    x_idx = [np.argmax(u_exp[:, 0, 0]) + i for i in delta_xs]
    for i in range(u_exp.shape[2]):
        for j in range(len(x_idx)):
            axs[0, 0].plot(
                time_mesh,
                error[x_idx[j], :, i],
                label=f"$\Delta$x = {delta_xs[j]}",
            )

    # plot percent_error vs spatial mesh
    # plot with space on the x-axis
    times = [0, 1, 5, 20, 40, 50, 100]
    for i in range(u_exp.shape[2]):
        for t in times:
            axs[0, 1].plot(
                spatial_mesh,
                error[:, t, i],
                label=f"t = {t}",
            )
    # plot total percent_error by species vs time step
    for i in range(u_exp.shape[2]):
        axs[1, 0].plot(
            time_mesh,
            total_error_over_time[:, i],
        )

    for i in range(u_exp.shape[2]):
        axs[1, 1].plot(
            spatial_mesh,
            total_error_over_space[:, i],
        )

    # Add legends
    axs[0, 1].legend(title="time step", loc="upper left")
    axs[0, 0].legend(title="steps from impulse", loc="upper right", ncol=2)

    # add letter labels for each fig
    titles = [
        "Error Over Time",
        "Error Over Space",
        "Total Error Over Time",
        "Total Error Over Space",
    ]
    xlabs = ["distance (um)", "time (usec)"]
    letters = ["A", "B", "C", "D"]
    ax = axs.flatten()
    for i in range(4):
        ax[i].annotate(
            letters[i],
            xy=(-0.1, 1.05),
            xycoords="axes fraction",
            fontsize=16,
            weight="bold",
        )

        # if i % 2 == 0:
        ax[i].set(xlabel=xlabs[i % 2], ylabel="error")
        ax[i].set_title(titles[i])

    # Set title and save
    fig.suptitle(
        "Percent Error Between Spectral and Finite Difference Diffusion Simulations",
        fontsize=18,
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}error.png", dpi=500)
    plt.show()


def plot_T(load_dir, save_dir, orientation="vertical"):
    print("Plotting T...")
    if orientation == "horizontal":
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    else:
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))

    labels_long = ["Calcium", "Calbindin", "Bound Calcium-Calbindin"]

    # Initialize simulation data
    T = np.load(load_dir)
    n_ca = np.max(T[:, 0, 0])
    n_calb = np.sum(T[:, 0, 1])
    total_particles = [
        n_ca,
        n_calb,
        n_calb,
    ]
    scaling_factor = [
        1,
        1.003387,
        4,
    ]
    spatial_mesh = np.linspace(0, 4, T.shape[0])
    time_mesh = [*range(0, T.shape[1])]
    n_eigenmodes = T.shape[0]

    print(T.shape)

    # plot with eigenmodes on the x-axis
    times = [0, 1, 5, 20, 40, 50, 100]
    for i in range(T.shape[2]):
        for t in times:
            if orientation == "horizontal":
                term_1, term_2 = 0, i
            else:
                term_1, term_2 = i, 0

            # TODO: determine if scaling is necessary
            axs[term_1, term_2].plot(
                [*range(0, n_eigenmodes)],
                T[:, t, i] / total_particles[i] / scaling_factor[i],
                label=f"t = {t}",
            )

    # plot with time on the x-axis
    modes = [0, 1, 2, 3, 4, 5, 10, 15, 19]
    for i in range(T.shape[2]):
        for m in modes:
            if orientation == "horizontal":
                term_1, term_2 = 1, i
            else:
                term_1, term_2 = i, 1

            axs[term_1, term_2].plot(
                time_mesh,
                T[m, :, i],
                label=f"m = {m}",
            )

    # Add labels
    xlabs = ["eigenmodes", "time (usec)"]
    for i in range(2):
        for j in range(T.shape[2]):
            if orientation == "horizontal":
                term_1, term_2 = i, j
            else:
                term_1, term_2 = j, i
            # if i == 0:
            #    axs[term_1, term_2].set_xlim(1.5, 3)
            axs[term_1, term_2].set(xlabel=xlabs[i], ylabel="temporal component, T(t)")
            axs[term_1, term_2].set_title(labels_long[j])

    # Set calbindin limits
    if orientation == "horizontal":
        term_1, term_2 = 0, 1
    else:
        term_1, term_2 = 1, 0
    """
    # axs[term_1, term_2].set_ylim([6.6225e-3, 6.643e-3])
    axs[term_1, term_2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    """

    # Add legends
    axs[term_1, term_2].legend(title="time step", loc="upper right")
    axs[1, 1].legend(title="eigenmode", loc="center", ncol=2)

    # add letter labels for each fig
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

    fig.suptitle("Spectral Calcium Diffusion with Calbindin Buffer", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{save_dir}T-{orientation}.png", dpi=500)
    # plt.show()


def plot_rxn_diffusion(load_dir, save_dir, orientation="vertical"):
    if orientation == "horizontal":
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    else:
        fig, axs = plt.subplots(3, 2, figsize=(10, 10))

    labels_long = ["Calcium", "Calbindin", "Bound Calcium-Calbindin"]

    # Initialize simulation data
    u = np.load(load_dir)
    n_ca = np.max(u[:, 0, 0])
    n_calb = np.sum(u[:, 0, 1])
    total_particles = [
        n_ca,
        n_calb,
        n_calb,
    ]
    scaling_factor = [
        1,
        1.003387,
        4,
    ]
    spatial_mesh = np.linspace(0, 4, u.shape[0])
    time_mesh = [*range(0, u.shape[1])]

    print(u.shape)

    # plot with space on the x-axis
    times = [0, 1, 5, 20, 40, 50, 100]
    for i in range(u.shape[2]):
        for t in times:
            if orientation == "horizontal":
                term_1, term_2 = 0, i
            else:
                term_1, term_2 = i, 0

            axs[term_1, term_2].plot(
                spatial_mesh,
                u[:, t, i] / total_particles[i] / scaling_factor[i],
                label=f"t = {t}",
            )

    # plot with time on the x-axis
    delta_xs = [*range(0, 10)]
    x_idx = [np.argmax(u[:, 0, 0]) + i for i in delta_xs]
    for i in range(u.shape[2]):
        for j in range(len(x_idx)):
            if orientation == "horizontal":
                term_1, term_2 = 1, i
            else:
                term_1, term_2 = i, 1

            axs[term_1, term_2].plot(
                time_mesh,
                u[x_idx[j], :, i] / total_particles[i] / scaling_factor[i],
                label=f"$\Delta$x = {delta_xs[j]}",
            )

    # Add labels
    xlabs = ["distance (um)", "time (usec)"]
    for i in range(2):
        for j in range(u.shape[2]):
            if orientation == "horizontal":
                term_1, term_2 = i, j
            else:
                term_1, term_2 = j, i
            if i == 0:
                axs[term_1, term_2].set_xlim(1.5, 3)
            axs[term_1, term_2].set(xlabel=xlabs[i], ylabel="Normalized particle count")
            axs[term_1, term_2].set_title(labels_long[j])

    # Set calbindin limits
    if orientation == "horizontal":
        term_1, term_2 = 0, 1
    else:
        term_1, term_2 = 1, 0
    # axs[term_1, term_2].set_ylim([6.6225e-3, 6.643e-3])
    axs[term_1, term_2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Add legends
    axs[0, 0].legend(title="time step", loc="upper left")
    axs[term_2, term_1].legend(title="steps from impulse", loc="upper right", ncol=2)

    # add letter labels for each fig
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
    fig.suptitle("Spectral Calcium Diffusion with Calbindin Buffer", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{save_dir}u-{orientation}.png", dpi=500)
    # plt.show()
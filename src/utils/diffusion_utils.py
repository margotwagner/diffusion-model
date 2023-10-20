import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_difference(true_load_dir, exp_load_dir, percent):
    # Initialize simulation data
    u_true = np.load(true_load_dir)
    u_exp = np.load(exp_load_dir)

    n_ca_exp = np.max(u_exp[:, 0, 0])
    n_calb_exp = np.sum(u_exp[:, 0, 1])

    n_ca_true = np.max(u_true[:, 0, 0])
    n_calb_true = np.sum(u_true[:, 0, 1])

    total_particles_exp = [
        n_ca_exp,
        n_calb_exp,
        n_calb_exp,
    ]
    total_particles_true = [
        n_ca_true,
        n_calb_true,
        n_calb_true,
    ]
    scaling_factor_exp = [
        1,
        1.003387,
        4,
    ]

    # scale results if necessary
    for species in range(len(total_particles_exp)):
        u_exp[:, :, species] = (
            u_exp[:, :, species]
            / total_particles_exp[species]
            / scaling_factor_exp[species]
        )

        # TODO: scale finite diff results
        u_true[:, :, species] = u_true[:, :, species] / total_particles_true[species]

    # get percent difference for each space/time step
    difference = np.abs(u_true - u_exp) / 0.5 * (np.abs(u_true) + np.abs(u_exp))

    if percent:
        return 100 * difference
    else:
        return difference


def diff_by_species(true_load_dir, exp_load_dir, percent):
    diff = get_difference(true_load_dir, exp_load_dir, percent=False)

    total_error_by_species = np.sum(diff, axis=(0, 1))

    if percent:
        return 100 * total_error_by_species
    else:
        return total_error_by_species


def diff_over_time(true_load_dir, exp_load_dir, percent):
    diff = get_difference(true_load_dir, exp_load_dir, percent=False)

    total_error_over_time = np.sum(diff, axis=0)

    if percent:
        return 100 * total_error_over_time
    else:
        return total_error_over_time


def diff_over_space(true_load_dir, exp_load_dir, percent):

    diff = get_difference(true_load_dir, exp_load_dir, percent=False)

    total_error_over_space = np.sum(diff, axis=1)

    if percent:
        return 100 * total_error_over_space
    else:
        return total_error_over_space


def total_diff(true_load_dir, exp_load_dir, percent):
    species_diff = diff_by_species(true_load_dir, exp_load_dir, percent=False)

    total_diff = np.sum(species_diff)

    if percent:
        return 100 * total_diff
    else:
        return total_diff

def ca_percent_error(true_load_dir, exp_load_dir, save_dir):
    # Initialize simulation data
    u_true = np.load(true_load_dir)
    u_exp = np.load(exp_load_dir)

    spatial_mesh = np.linspace(0, 4, u_true.shape[0])
    time_mesh = [*range(0, u_true.shape[1])]

    # get percent difference for each space/time step
    percent_diff = get_difference(true_load_dir, exp_load_dir, percent=True)

    # get total percent difference for each species
    total_error_by_species = diff_by_species(true_load_dir, exp_load_dir, percent=True)
    total_error_over_time = diff_over_time(true_load_dir, exp_load_dir, percent=True)
    total_error_over_space = diff_over_space(true_load_dir, exp_load_dir, percent=True)
    total_error = total_diff(true_load_dir, exp_load_dir, percent=True)

    print("Total error: ", total_error)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # plot total percent_error by species vs time step
    species = ["ca", "calb", "ca-calb"]

    # plot percent_error vs time step
    # plot with time on the x-axis
    delta_xs = [*range(0, 5)]
    x_idx = [np.argmax(u_exp[:, 0, 0]) + i for i in delta_xs]
    colors = sns.color_palette("flare", len(x_idx))
    for x in range(len(x_idx)):
        axs[0].plot(
            time_mesh,
            percent_diff[x_idx[x], :, 0],
            label=f"{delta_xs[x]}",
            color=colors[x],
            linewidth=2.5,
            alpha=0.8,
        )

    # plot percent_error vs spatial mesh
    # plot with space on the x-axis
    times = [0, 1, 2, 3, 4, 5, 10] #20, 40, 50, 100]
    colors = sns.color_palette("crest", len(times))
    for t in range(len(times)):
        axs[1].plot(
            spatial_mesh,
            percent_diff[:, t, 0],
            label=f"{times[t]}",
            color=colors[t],
            linewidth=2.5,
            alpha=0.8,
        )

    # Add legends
    axs[1].legend(title="time step", loc="upper left", fontsize=10)
    axs[0].legend(title="$\Delta$x", loc="upper right", fontsize=10)

    # Set limits
    # axs[0, 0].set_xlim([0, 40])
    axs[1].set_xlim([1.5, 3.5])

    # add letter labels for each fig
    titles = [
        "Calcium Difference Over Time",
        "Calcium Difference Over Space",
    ]
    xlabs = ["time (usec)", "distance (um)"]
    ax = axs.flatten()
    for i in range(2):

        if i % 2 == 0:
            ax[i].set_xlim([-.05, 8])
        else:
            ax[i].set_xlim([2, 2.8])

        ax[i].set_xlabel(xlabs[i % 2], fontsize=16)
        ax[i].set_ylabel("% difference", fontsize=16)
        ax[i].set_title(titles[i], fontsize=18)
        ax[i].tick_params(axis='both', which='major', labelsize=14)
        ax[i].tick_params(axis='both', which='minor', labelsize=14)

    # print summary report
    print("Percent Difference Summary Report")
    print(35 * "-")
    print(f"Total Difference: {total_error}")
    print("Total Difference by Species:\n")
    for i in range(len(species)):
        print(f"{species[i]}:\t {total_error_by_species[i]}")

    # Set title and save
    '''
    fig.suptitle(
        "Difference Between Spectral and Finite Difference Diffusion Simulations",
        fontsize=20,
    )'''
    plt.tight_layout()
    plt.savefig(f"{save_dir}error.png", dpi=500)
    plt.show()


def ca_percent_error(true_load_dir, exp_load_dir, save_dir):
    # Initialize simulation data
    u_true = np.load(true_load_dir)
    u_exp = np.load(exp_load_dir)

    spatial_mesh = np.linspace(0, 4, u_true.shape[0])
    time_mesh = [*range(0, u_true.shape[1])]

    # get percent difference for each space/time step
    percent_diff = get_difference(true_load_dir, exp_load_dir, percent=True)

    # get total percent difference for each species
    total_error_by_species = diff_by_species(true_load_dir, exp_load_dir, percent=True)
    total_error_over_time = diff_over_time(true_load_dir, exp_load_dir, percent=True)
    total_error_over_space = diff_over_space(true_load_dir, exp_load_dir, percent=True)
    total_error = total_diff(true_load_dir, exp_load_dir, percent=True)
    
    print(total_error_by_species[0])

    print("Total error: ", total_error)

    '''
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # plot total percent_error by species vs time step
    species = ["ca", "calb", "ca-calb"]

    # plot percent_error vs time step
    # plot with time on the x-axis
    delta_xs = [*range(0, 5)]
    x_idx = [np.argmax(u_exp[:, 0, 0]) + i for i in delta_xs]
    colors = sns.color_palette("flare", len(x_idx))
    for x in range(len(x_idx)):
        axs[0].plot(
            time_mesh,
            percent_diff[x_idx[x], :, 0],
            label=f"{delta_xs[x]}",
            color=colors[x],
            linewidth=2.5,
            alpha=0.8,
        )

    # plot percent_error vs spatial mesh
    # plot with space on the x-axis
    times = [0, 1, 2, 3, 4, 5, 10] #20, 40, 50, 100]
    colors = sns.color_palette("crest", len(times))
    for t in range(len(times)):
        axs[1].plot(
            spatial_mesh,
            percent_diff[:, t, 0],
            label=f"{times[t]}",
            color=colors[t],
            linewidth=2.5,
            alpha=0.8,
        )

    # Add legends
    axs[1].legend(title="time step", loc="upper left", fontsize=10)
    axs[0].legend(title="$\Delta$x", loc="upper right", fontsize=10)

    # Set limits
    # axs[0, 0].set_xlim([0, 40])
    axs[1].set_xlim([1.5, 3.5])

    # add letter labels for each fig
    titles = [
        "Calcium Difference Over Time",
        "Calcium Difference Over Space",
    ]
    xlabs = ["time (usec)", "distance (um)"]
    ax = axs.flatten()
    for i in range(2):

        if i % 2 == 0:
            ax[i].set_xlim([-.05, 8])
        else:
            ax[i].set_xlim([2, 2.8])

        ax[i].set_xlabel(xlabs[i % 2], fontsize=16)
        ax[i].set_ylabel("% difference", fontsize=16)
        ax[i].set_title(titles[i], fontsize=18)
        ax[i].tick_params(axis='both', which='major', labelsize=14)
        ax[i].tick_params(axis='both', which='minor', labelsize=14)

    # print summary report
    print("Percent Difference Summary Report")
    print(35 * "-")
    print(f"Total Difference: {total_error}")
    print("Total Difference by Species:\n")
    for i in range(len(species)):
        print(f"{species[i]}:\t {total_error_by_species[i]}")

    # Set title and save

    fig.suptitle(
        "Difference Between Spectral and Finite Difference Diffusion Simulations",
        fontsize=20,
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}error.png", dpi=500)
    plt.show()
    '''





def percent_error(true_load_dir, exp_load_dir, save_dir):
    # Initialize simulation data
    u_true = np.load(true_load_dir)
    u_exp = np.load(exp_load_dir)

    spatial_mesh = np.linspace(0, 4, u_true.shape[0])
    time_mesh = [*range(0, u_true.shape[1])]

    # get percent difference for each space/time step
    percent_diff = get_difference(true_load_dir, exp_load_dir, percent=True)

    # get total percent difference for each species
    total_error_by_species = diff_by_species(true_load_dir, exp_load_dir, percent=True)
    total_error_over_time = diff_over_time(true_load_dir, exp_load_dir, percent=True)
    total_error_over_space = diff_over_space(true_load_dir, exp_load_dir, percent=True)
    total_error = total_diff(true_load_dir, exp_load_dir, percent=True)

    print("Total error: ", total_error)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # plot total percent_error by species vs time step
    species = ["ca", "calb", "ca-calb"]
    for i in range(u_exp.shape[2]):
        axs[0, 0].plot(
            time_mesh,
            total_error_over_time[:, i],
            label=species[i],
        )

    for i in range(u_exp.shape[2]):
        axs[0, 1].plot(
            spatial_mesh,
            total_error_over_space[:, i],
            label=species[i],
        )

    # plot percent_error vs time step
    # plot with time on the x-axis
    delta_xs = [*range(0, 10)]
    x_idx = [np.argmax(u_exp[:, 0, 0]) + i for i in delta_xs]
    for j in range(len(x_idx)):
        axs[1, 0].plot(
            time_mesh,
            percent_diff[x_idx[j], :, 0],
            label=f"$\Delta$x = {delta_xs[j]}",
        )

    # plot percent_error vs spatial mesh
    # plot with space on the x-axis
    times = [0, 1, 5, 20, 40, 50, 100]
    for t in times:
        axs[1, 1].plot(
            spatial_mesh,
            percent_diff[:, t, 0],
            label=f"t = {t}",
        )

    # Add legends
    axs[0, 0].legend(title="species", loc="upper right")
    axs[0, 1].legend(title="species", loc="upper right")
    axs[1, 1].legend(title="time step", loc="upper left")
    axs[1, 0].legend(title="steps from impulse", loc="upper right", ncol=2)

    # Set limits
    # axs[0, 0].set_xlim([0, 40])
    axs[0, 1].set_xlim([1.5, 3.5])
    axs[1, 1].set_xlim([1.5, 3.5])

    # add letter labels for each fig
    titles = [
        "Total Difference Over Time",
        "Total Difference Over Space",
        "Calcium Difference Over Time",
        "Calcium Difference Over Space",
    ]
    xlabs = ["time (usec)", "distance (um)"]
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

        if i % 2 == 0:
            ax[i].set_xlim([-1, 20])
        else:
            ax[i].set_xlim([1.5, 3.5])

        ax[i].set(xlabel=xlabs[i % 2], ylabel="% difference")
        ax[i].set_title(titles[i])

    # print summary report
    print("Percent Difference Summary Report")
    print(35 * "-")
    print(f"Total Difference: {total_error}")
    print("Total Difference by Species:\n")
    for i in range(len(species)):
        print(f"{species[i]}:\t {total_error_by_species[i]}")

    # Set title and save
    fig.suptitle(
        "Difference Between Spectral and Finite Difference Diffusion Simulations",
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


def plot_calcium(load_dir, save_dir):
    fig = plt.figure(figsize=(6, 5))

    # Initialize simulation data
    u = np.load(load_dir)
    n_ca = np.max(u[:, 0, 0])

    spatial_mesh = np.linspace(0, 4, u.shape[0])

    colors = sns.color_palette("hls", 10)
    # colors.reverse()

    # plot with space on the x-axis
    times = [0, 1, 2, 3, 5, 20, 50, 100]
    # times.reverse()
    for i in range(len(times)):
        plt.plot(
            spatial_mesh,
            u[:, times[i], 0] / n_ca,
            label=f"{times[i]}",
            color=colors[i],
            linewidth=2,
            alpha=0.8,
        )

    # Add legends
    plt.legend(title="time step", loc="upper left")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(1.8, 2.8)
    plt.xlabel("distance (um)", fontsize=14)

    # Set title and save
    plt.savefig(f"{save_dir}calcium.png", dpi=500)
    plt.show()


def plot_calbindin(load_dir, save_dir):
    fig = plt.figure(figsize=(6, 5))

    # Initialize simulation data
    u = np.load(load_dir)
    n_calb = np.sum(u[:, 0, 1])

    scaling_factor = 1.003387

    spatial_mesh = np.linspace(0, 4, u.shape[0])

    colors = sns.color_palette("hls", 10)
    # colors.reverse()

    # plot with space on the x-axis
    times = [0, 1, 2, 3, 5, 20, 50, 100]
    # times.reverse()
    for i in range(len(times)):
        plt.plot(
            spatial_mesh,
            u[:, times[i], 1] / n_calb,  # / (1 + 1.2e-3),
            label=f"{times[i]}",
            color=colors[i],
            linewidth=2,
            alpha=0.8,
        )

    # Add legends
    plt.legend(title="time step", loc="lower left")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # plt.ylim(6.645e-3, 6.670e-3)
    plt.ylim(6.57e-3, 6.68e-3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(1.7, 3.0)
    plt.xlabel("distance (um)", fontsize=14)

    # Set title and save
    plt.savefig(f"{save_dir}calbindin.png", dpi=500)
    plt.show()


def plot_ca_calbindin(load_dir, save_dir):
    fig = plt.figure(figsize=(6, 5))

    # Initialize simulation data
    u = np.load(load_dir)
    n_calb = np.sum(u[:, 0, 1])

    scaling_factor = 1.003387

    spatial_mesh = np.linspace(0, 4, u.shape[0])

    colors = sns.color_palette("hls", 10)
    # colors.reverse()

    # plot with space on the x-axis
    times = [0, 1, 2, 3, 5, 20, 50, 100]
    # times.reverse()
    for i in range(len(times)):
        plt.plot(
            spatial_mesh,
            u[:, times[i], 2] / n_calb,
            label=f"{times[i]}",
            color=colors[i],
            linewidth=2,
            alpha=0.8,
        )

    # Add legends
    plt.legend(title="time step", loc="upper left")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # plt.ylim(6.645e-3, 6.670e-3)
    # plt.ylim(6.57e-3, 6.68e-3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(1.7, 3.0)
    plt.xlabel("distance (um)", fontsize=14)

    # Set title and save
    plt.savefig(f"{save_dir}ca-calbindin.png", dpi=500)
    plt.show()

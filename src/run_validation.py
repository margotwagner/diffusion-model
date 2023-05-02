import RandomWalk as rw
import EigenmarkovDiffusion as emd


# TODO: check for latching of param values inside called functions
def run_validation(
    n_particles=50,  # number of molecules
    n_spatial_locs=11,  # define number of grid points along 1D line
    n_time_pts=1000,  # number of time points
    particle_start_loc=5,  # starting position of input impulse for molecules
    binomial_sampling=False,  # use binomial sampling for eme markov simulation
    print_eigenvalues_and_vectors=False,
    print_eigenmode_init_conditions=False,
    print_eigenmode_transition_probability=False,
    print_eigenmodes_to_spatial_nodes=False,
    plot_random_walk=False,
    plot_eigenmodes=False,
    plot_eigenvectors=False,
    plot_eigenmode_init_conditions=False,
    plot_eigenmode_markov_simulation=False,
):
    print("Initializing random walk and eigenmarkov diffusion simulations.")
    random_walk = rw.RandomWalk(
        n_particles=n_particles,
        n_spatial_locs=n_spatial_locs,
        n_time_pts=n_time_pts,
        particle_start_loc=particle_start_loc,
    )

    eigenmarkov = emd.EigenmarkovDiffusion(
        n_particles=n_particles,
        n_spatial_locs=n_spatial_locs,
        n_time_pts=n_time_pts,
        particle_start_loc=particle_start_loc,
        scaling_factor=2,
    )

    # run random walk simulation and plot output
    # random_walk.draw_impulse()

    particle_locs = random_walk.run_simulation()

    print("Beginning random walk simulation.")
    unnorm_n_per_loc, n_per_loc, mean_n_per_loc = random_walk.postprocess_run(
        particle_locs, plot=plot_random_walk
    )
    print("Finished random walk simulation.")

    print("Beginning eigenmarkov simulation.")
    # run eigenmarkov simulation and plot output
    n_per_eigenmode_state = eigenmarkov.run_simulation(
        binomial_sampling=binomial_sampling,
        print_eigenvalues_and_vectors=print_eigenvalues_and_vectors,
        print_init_conditions=print_eigenmode_init_conditions,
        print_transition_probability=print_eigenmode_transition_probability,
        plot_eigenvectors=plot_eigenvectors,
        plot_eigenmodes=plot_eigenmodes,
        plot_init_conditions=plot_eigenmode_init_conditions,
        plot_simulation=plot_eigenmode_markov_simulation,
    )
    print("Finished eigenmarkov simulation.")

    print("Converting eigenmodes to spatial nodes.")
    node_vals_from_modes = eigenmarkov.convert_to_spatial_nodes(
        n_per_eigenmode_state, print_output=print_eigenmodes_to_spatial_nodes
    )


if __name__ == "__main__":
    run_validation(
        n_time_pts=1000,
        binomial_sampling=False,
        print_eigenvalues_and_vectors=False,
        print_eigenmode_init_conditions=False,
        print_eigenmode_transition_probability=False,
        print_eigenmodes_to_spatial_nodes=False,
        plot_random_walk=False,
        plot_eigenmodes=False,
        plot_eigenvectors=False,
        plot_eigenmode_init_conditions=False,
        plot_eigenmode_markov_simulation=True,
    )

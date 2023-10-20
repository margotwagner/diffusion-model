import sys
sys.path.append("../../src/")
import models.RandomWalk as rw
import models.EigenmarkovDiffusion as emd


def run_validation(
    n_particles,  # number of molecules
    n_spatial_locs,  # define number of grid points along 1D line
    n_time_pts,  # number of time points
    particle_start_loc,  # starting position of input impulse for molecules
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
    random_walk.draw_impulse()

    particle_locs = random_walk.run_simulation()

    unnorm_n_per_loc, n_per_loc, mean_n_per_loc = random_walk.postprocess_run(
        particle_locs, plot=plot_random_walk
    )

    # run eigenmarkov simulation and plot output
    n_per_eigenmode_state = eigenmarkov.run_simulation(
        binomial_sampling=binomial_sampling,  # use binomial sampling eme markov simulation
        print_eigenvalues_and_vectors=print_eigenvalues_and_vectors,
        print_init_conditions=print_eigenmode_init_conditions,
        print_transition_probability=print_eigenmode_transition_probability,
        plot_eigenvectors=plot_eigenvectors,
        plot_eigenmodes=plot_eigenmodes,
        plot_init_conditions=plot_eigenmode_init_conditions,
        plot_simulation=plot_eigenmode_markov_simulation,
    )
    node_vals_from_modes = eigenmarkov.convert_to_spatial_nodes(
        n_per_eigenmode_state, print_output=print_eigenmodes_to_spatial_nodes
    )


if __name__ == "__main__":
    run_validation(
        n_particles=50,  # number of molecules
        n_spatial_locs=101,  # define number of grid points along 1D line
        particle_start_loc=58,  # starting position of input impulse for molecules
        n_time_pts=100,  # number of time points
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
    )

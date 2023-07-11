import PlotMultiruns as pm

plotter = pm.PlotMultiRuns(
    rw_dir= "/Users/margotwagner/diffusion-model/data/eme-validation/random-walk/20230711_163728/rw-run-{}.csv",  #"/Users/margotwagner/diffusion-model/data/eme-validation/random-walk/20230118_192600/rw-run-{}.csv",
    eme_dir= "/Users/margotwagner/diffusion-model/data/eme-validation/markov-eme/20230711_163210/eme-run-{}.csv",  #"/Users/margotwagner/diffusion-model/data/eme-validation/markov-eme/20230405_105433_2/eme-run-{}.csv",
    n_runs=100,
    n_spatial_locs=101,
    n_time_pts=1000,
    particle_start_loc=58,
    n_particles=50,
    plot_eme=True,
    plot_rw=True,
)

plotter.plot_separately()

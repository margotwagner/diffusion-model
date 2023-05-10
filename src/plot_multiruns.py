import PlotMultiruns as pm

plotter = pm.PlotMultiRuns(
    rw_dir="/Users/margotwagner/diffusion-model/data/eme-validation/random-walk/20230118_192600/rw-run-{}.csv",
    eme_dir="/Users/margotwagner/diffusion-model/data/eme-validation/markov-eme/20230405_105433_2/eme-run-{}.csv",
    # 105433_2
    # 103751_sqrt2
    n_runs=1000,
    n_spatial_locs=11,
    n_time_pts=1000,
    particle_start_loc=5,
    n_particles=50,
    plot_eme=True,
    plot_rw=True,
)

plotter.plot_separately()

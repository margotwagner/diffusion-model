import utils.PlotMultiruns as pm

PATH = "20231105_230244"

plotter = pm.PlotMultiRuns(
    rw_dir="../data/eme-validation/random-walk/20231105_231949/rw-run-{}.csv",  # "/Users/margotwagner/diffusion-model/data/eme-validation/random-walk/20230118_192600/rw-run-{}.csv",
    eme_dir="../data/eme-validation/markov-eme/20231105_231949/eme-run-{}.csv",  # "/Users/margotwagner/diffusion-model/data/eme-validation/markov-eme/20230405_105433_2/eme-run-{}.csv",
    n_runs=10,
    n_spatial_locs=101,
    n_time_pts=100,  # TODO: make automatic
    particle_start_loc=58,
    n_particles=50,
    plot_eme=True,  # TODO: fix LHS entry
    plot_rw=False,
)

# plotter.plot_multiruns()
# TODO: debug multirun plot for EME -- giving impulse at boundary
# plotter.plot_multiruns_time([0, 99, 99])
plotter.plot_multiruns()

import utils.PlotMultiruns as pm

PATH = "20231112_233047"

def get_ca_init_loc():
    # VDCC
    vdcc_loc = 2.35  # um

    return vdcc_loc


def get_ca_init_idx(n_space_pts):
    vdcc_loc = get_ca_init_loc()

    z = 4  # um
    dx = z / n_space_pts

    return int(vdcc_loc / dx)

n_time_pts = 100  # number of time points
n_space_pts = 101  # 150  # number of spatial points
ca_init_idx = get_ca_init_idx(n_space_pts)

plotter = pm.PlotMultiRuns(
    rw_dir="../data/eme-validation/random-walk/20231112_234059/rw-run-{}.csv",  # "/Users/margotwagner/diffusion-model/data/eme-validation/random-walk/20230118_192600/rw-run-{}.csv",
    eme_dir="../data/eme-validation/markov-eme/20240224_141238/eme-run-{}.csv",  # "/Users/margotwagner/diffusion-model/data/eme-validation/markov-eme/20230405_105433_2/eme-run-{}.csv",
    n_runs=10,
    n_spatial_locs=n_space_pts,
    n_time_pts=n_time_pts,  # TODO: make automatic
    particle_start_loc=ca_init_idx,
    n_particles=50,
    plot_eme=True,  # TODO: fix LHS entry
    plot_rw=False,
)

# plotter.plot_multiruns()
# TODO: debug multirun plot for EME -- giving impulse at boundary
# plotter.plot_multiruns_time([0, 99, 99])
plotter.plot_multiruns()

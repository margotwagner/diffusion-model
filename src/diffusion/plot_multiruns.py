"""Plot saved output from run_multiruns.py -- multiple diffusion simulations of either random walk or EME method.

Usage: python3 ./plot_multiruns.py
"""

__author__ = ["Margot Wagner"]
__contact__ = "mwagner@ucsd.edu"
__date__ = "2023/06/13"

import sys
sys.path.append("../../src/")
import utils.PlotMultiruns as pm

plotter = pm.PlotMultiRuns(
    eme_dir="/Users/margotwagner/projects/diffusion-model/data/eme-validation/markov-eme/20231020_140513/", #eme-run-{}.csv",
    rw_dir="/Users/margotwagner/projects/diffusion-model/data/eme-validation/random-walk/20231020_140528/",  # rw-run-{}.csv",
    plot_eme=True,  # TODO: fix LHS entry
    plot_rw=True,
)

# plotter.plot_multiruns()
# TODO: debug multirun plot for EME -- giving impulse at boundary
plotter.plot_multiruns_time([0, 10, 99])

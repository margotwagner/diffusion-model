'''
Script for analyzing MCell molecule location data for calcium diffusion

 '''

# import packages
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from scipy import stats


def time_hist(dir,iter_num, plot=True):
    '''
    Calculates concentration of calcium per spherical shell at a given radius
    over time for MCell simulation.
    :param dir: MCell ASCII data location
    :param iter_num: which iteration/time point to plot
    :param plot: if output should be plotted
    :return: ca radial locations
    '''


    ca_hist = []
    for seed in sorted(os.listdir(mcell_viz_dir)):
        iter_file = os.path.join(mcell_viz_dir, seed, 'Scene.ascii.{}.dat'.format(iter_num))
        loc_data = pd.read_csv(iter_file, delim_whitespace=True, header=None,
                           names=['type', 'id', 'x', 'y', 'z', 'norm_x', 'norm_y', 'norm_z'])
        # select only calcium
        loc_data = loc_data[loc_data['type'] == 'ca']

        # radius from x, y, and z coordinates
        loc_data['r'] = np.sqrt(loc_data['x'] ** 2 + loc_data['y'] ** 2 + loc_data['z'] ** 2)

        for value in loc_data['r'].values:
            ca_hist.append(value)

    if plot:
        #plt.hist(ca_hist, bins=250, color='C0')
        sns.distplot(ca_hist, bins=100)
        plt.show()

    return ca_hist

# plot for multiple time points
#mcell_viz_dir = "/Users/margotwagner/projects/mcell/simple_geom/" \
#               "infinite_space/half_space_plane_files/mcell/output_data/viz_data_ascii"

mcell_viz_dir = "/Users/margotwagner/projects/mcell/simple_geom/model_1/" \
                "model_1_ca_diff_files/mcell/output_data/viz_data_ascii"

nums = ['00050', '00100', '00500','05000']
#nums = ['00500','01910','04280', '10000']

data = []
for num in nums:
    ca_hist = time_hist(mcell_viz_dir, num, plot=False)
    sns.distplot(ca_hist, label=num, bins=100)

plt.xlabel('Radius')
plt.ylabel('Frequency')
plt.title('Calcium frequency with radius')
plt.legend()
plt.savefig('mcell_ca_impulse_diff.png')
plt.show()




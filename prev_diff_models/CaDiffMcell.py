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
    rxn_rad = 0.001     # reaction radius size
    snare_locs = [(0, 0),
                  (0.035, 0),
                  (-0.035, 0),
                  (0.0175, 0.0303),
                  (-0.0175, 0.0303),
                  (0.0175, -0.0303),
                  (-0.0175, -0.0303)]      # SNARE (x,y) locations
    z = -0.25

    ca_hist = []
    for seed in sorted(os.listdir(mcell_viz_dir)):


        # data file location
        iter_file = os.path.join(mcell_viz_dir, seed, 'Scene.ascii.{}.dat'.format(iter_num))

        # create dataframe
        loc_data = pd.read_csv(iter_file, delim_whitespace=True, header=None,
                           names=['type', 'id', 'x', 'y', 'z', 'norm_x', 'norm_y', 'norm_z'])
        # select only calcium
        loc_data = loc_data[loc_data['type'] == 'ca']

        # radius from x, y, and z coordinates
        loc_data['r'] = np.sqrt(loc_data['x'] ** 2 + loc_data['y'] ** 2 + loc_data['z'] ** 2)

        snare_ca_tot = 0        # number of calcium in SNARE rxn zone
        for x,y in snare_locs:
            print()
            print(x,y)
            for ca_x, ca_y, ca_z in zip(loc_data['x'], loc_data['y'], loc_data['z']):
                if ca_z <= z + rxn_rad:
                    print(ca_x, ca_y, ca_z)

            snare_ca_tot += len(loc_data[(loc_data['x'] >= x - rxn_rad) & (loc_data['x'] <= x + rxn_rad) 
                    & (loc_data['y'] >= y - rxn_rad) & (loc_data['y'] <= y + rxn_rad) 
                    & (loc_data['z'] <= z + rxn_rad)])

        print(snare_ca_tot)

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

#mcell_viz_dir = "/Users/margotwagner/projects/mcell/simple_geom/model_1/" \
#                "model_1_ca_diff_files/mcell/output_data/viz_data_ascii"

mcell_viz_dir = "/Users/margotwagner/ucsd/research/DiffusionModel/" \
                "rect_zeroflux_nocalbpmca_files/mcell/output_data/viz_data_ascii"



#nums = ['0050', '0100', '0500','1000']
nums = ['0010']
#nums = ['00500','01910','04280', '10000']

data = []
for num in nums:
    ca_hist = time_hist(mcell_viz_dir, num, plot=False)
    #sns.distplot(ca_hist, label=num, bins=100)

'''
plt.xlabel('Radius')
plt.ylabel('Frequency')
plt.title('Calcium frequency with radius')
plt.legend()
plt.savefig('mcell_ca_impulse_diff.png')
plt.show()
'''




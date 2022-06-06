#!PYTHONPATH

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from scipy import stats
from math import pi

def ca_snare_finder(mcell_viz_dir, timepoint):
    '''
    Calculates average and standard deviation of concentration of calcium across all seeds
    in a box around the SNARE complexes at a given time for MCell simulation.

    :param mcell_dir: MCell ASCII data location
    :param timepoint: which iteration/time point to plot
    :param plot: if output should be plotted

    :return: concentration of Ca in the SNARE at the given timepoint
    '''   
    
    # SNARE box dimensions (using refrac box dimensions)
    delta_x = 0.040   # SNARE BOX X LEN (um)
    delta_y = 0.035   # SNARE BOX Y LEN (um)
    delta_z = 0.05    # SNARE BOX X LEN (um)

    # SNARE BOX LOCATION
    # for a giant box snare complex 
    z_s = -.24550 # using refrac box location
    snare_locs = [(0, 0, z_s)] # for only one giant SNARE (x, y, z)

    conc_ca_M_all_seeds = []

    # for each seed, find the concentration of Ca2+ in the snare box at timepoint
    for seed in sorted(os.listdir(mcell_viz_dir)):

        # data file location
        seed_data_dir = os.path.join(mcell_viz_dir, seed, 'Scene.ascii.{}.dat'.format(timepoint))

        # create dataframe
        loc_data = pd.read_csv(seed_data_dir, delim_whitespace=True, header=None,
                           names=['type', 'id', 'x', 'y', 'z', 'norm_x', 'norm_y', 'norm_z'])

        # select only calcium
        loc_data = loc_data[loc_data['type'] == 'ca']

        # initialize number of calcium in SNARE rxn zone
        snare_ca_tot = 0        

        # ONLY 1 SNARE TO START (big snare box)
        for x_s, y_s, z_s in snare_locs:

            # Ugly method (TODO: make nicer)
            # get calcium location (x,y,z)
            for ca_x, ca_y, ca_z in zip(loc_data['x'], loc_data['y'], loc_data['z']):
                # check z direction
                if (ca_z >= z_s - (delta_z / 2)) & (ca_z <= z_s + (delta_z / 2)):

                    # check x direction
                    if (ca_x >= x_s - (delta_x / 2)) & (ca_x <= x_s + (delta_x / 2)):

                        # check y direction
                        if (ca_y >= y_s - (delta_y / 2)) & (ca_y <= y_s +  (delta_y / 2)):
                            snare_ca_tot += 1

        # summary 
        if snare_ca_tot != 0:
            # convert to concentration (# ca/um^3)
            conc_ca = snare_ca_tot/ (delta_x * delta_y * delta_z)

            conc_ca_M = conc_ca / (1e-15 * 6.022e23) # convert to Molar
        
        else:
            conc_ca_M = 0

        # append seed conc to all seeds
        conc_ca_M_all_seeds.append(conc_ca_M)

    return np.mean(conc_ca_M_all_seeds), np.std(conc_ca_M_all_seeds)


# plot for multiple time points
mcell_dir = "/home/bartol/mcell_projects_2/margot/rect_zeroflux_nocalbpmca_files/" \
            "mcell/output_data/viz_data_ascii"

# get time points
timepts = [scene.split('.')[2] for scene in os.listdir(os.path.join(mcell_dir, 'seed_00001'))]
timepts.sort()

#sample points
sampled_timepts = [timepts[i] for i in range(len(timepts)) if i%1==0]

# All time points
ca_conc_all_mean = []
ca_conc_all_std = []
for tp in sampled_timepts:
    ca_conc_all_mean.append(ca_snare_finder(mcell_dir, tp)[0])
    ca_conc_all_std.append(ca_snare_finder(mcell_dir, tp)[1])

with open("/home/bartol/mcell_projects_2/margot/ca_conc_all_std.txt", "w") as output:
    output.write(str(ca_conc_all_std))

with open("/home/bartol/mcell_projects_2/margot/ca_conc_all_mean.txt", "w") as output:
    output.write(str(ca_conc_all_mean))

timepts_int = [int(tp) for tp in sampled_timepts]

plt.figure(figsize=(7, 4))
plt.fill_between(timepts_int, np.array(ca_conc_all_mean) - np.array(ca_conc_all_std), np.array(ca_conc_all_mean) + np.array(ca_conc_all_std),
                alpha = 0.2, color = 'k')
plt.plot(timepts_int, ca_conc_all_mean)
plt.plot(timepts_int, np.array(ca_conc_all_mean) + np.array(ca_conc_all_std), color = 'k', alpha = 0.5)
plt.plot(timepts_int, np.array(ca_conc_all_mean) - np.array(ca_conc_all_std), color = 'k', alpha = 0.5)
plt.xlabel("Timepoint (microseconds)")
plt.ylabel("Concentration of Ca in the SNARE box")
plt.savefig('/home/bartol/mcell_projects_2/margot/ca_diff_mcell_output.png')

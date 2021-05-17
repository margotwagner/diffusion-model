'''
Script modeling calcium diffusion as a series of Gaussians where diffusion is
infinite in the x-direction and semi-infinite in the y- and z-direction. No
other molecules are included, only calcium.
'''
import numpy as np
import matplotlib.pyplot as plt

def gaussian_series(r, t, x0, t0=0, impulse=1, y_terms=(-2, 2), z_terms=(-1,1)):
    '''
    Series of gaussians to summarize infinite diffusion in x-direction and
     semi-infinite diffusion in y- and z-direction with i terms in y direction
      and j terms in z direction.
      y terms are spaced by one width and z terms are 2w with additional 2
       factor term.
       Describes the concentration of calcium at a given radius and time.
    :param r: radius of interest for calcium quantity (um)
    :param t: time point for calcium quantity (ms)
    :param x0: distance from center (vdcc to snare distance)
    :param t0: time of impulse
    :impulse: quantity release at time t0
    :param y_terms: number for y terms to include
    :param z_terms: number of z terms to include
    :return:
    '''
    # DEFINE CONSTANTS
    # calcium diffusion constant
    d_cm2_s = 2.2e-6  # cm^2/s
    diff_const = d_cm2_s * ((10 ** 6) ** 2) / 1000 / (100 ** 2)  # um^2/ms

    # box width
    w = 0.5  # width of box in y/z dimensions (um)

    #const
    reflection_factor = 2       # accounts for reflection on z-axis
    diff_3d = 3 / 2  # accounts for 3d diffusion in normalization term

    # Constant term outside of series
    input = reflection_factor*impulse
    norm_term = (4 * np.pi * diff_const * (t - t0)) ** diff_3d
    const_term = input / norm_term

    # unpack tuple input
    y_start, y_stop = y_terms
    z_start, z_stop = z_terms

    # SERIES TERM
    series_term = 0
    for i in range(y_start, y_stop):
        for j in range(z_start, z_stop):
            # FIX THIS TO BE D = SQRT((X - X0)^2....)
            r0ij = np.sqrt((x0 ** 2) + (((i + 0.5) * w) ** 2) + ((2 * j * w) ** 2))
            exp_r0ij = np.exp(-((r - r0ij) ** 2) / (4 * diff_const * (t - t0)))
            series_term += exp_r0ij

    return const_term * series_term

#def gaussian_series(r, t, x0, t0=0, impulse=1, y_terms=1, z_terms=1):
# should properly define dirac
r, r_step = np.linspace(0, 4, 100, retstep=True)    # um
#t_range = np.linspace(0.5, 3, 5)    # ms
t_range = np.array([50, 100, 500, 5000])
t_range = t_range/1000
print(t_range)
#t = 1
x0 = 0.3       # distance of vdcc to snare (um)
t0 = 0         # ms
ca = 1

for t in t_range:
    amount_of_ca = (2 / 3) * np.pi * (r ** 2) * r_step * \
                   gaussian_series(r, t, x0, y_terms=(-200, 200), z_terms=(-100, 100))
    plt.plot(r, amount_of_ca, label="{:05d}".format(int(t*1000)))

plt.title('Calcium diffusion')
plt.xlabel('Radius (nm)')
plt.ylabel('Concentration of calcium')
plt.legend()
plt.savefig('gauss_ca_impulse_diff.png')
plt.show()

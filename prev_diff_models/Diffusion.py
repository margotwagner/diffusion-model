import numpy as np
import matplotlib.pyplot as plt

def diffusion_3d(r, t, diff_const, impulse=1, t0=0, r0=0):
    '''
    Gaussian diffusive kernel for 3d diffusion in an infinite half space, where
     reflection occurs.
     r: radius values of interest
     t: time values of interest
     diff_const: calcium diffusion constant
     impulse:   impulse injection at t0
     t0: time of impulse (relative time)
     r0: location of impulse (relative location)
    '''

    reflection_factor = 2    # accounts for reflection on z-axis
    diff_3d = 3/2            # accounts for this being 3d diffusion

    input = reflection_factor*impulse
    norm_term = (4 * np.pi * diff_const * (t - t0))**(diff_3d)
    exp_term = -((r - r0) ** 2) / (4 * diff_const * (t - t0))

    return (input / norm_term) * np.exp(exp_term)
    #return a * np.exp((-(x - b) ** 2) / c)

r, r_step = np.linspace(0, 4, 100, retstep=True)    # um
#print(r_step)
t_range = np.linspace(0.6, 5)    # ms
#print(t_range)
d_cm2_s = 2.2e-6    # cm^2/s
d_um2_ms = d_cm2_s * ((10**6)**2) / 1000 / (100**2)    # um^2/ms
r0 = 0.21    # um
t0 = 0       # ms
ca = 1    # number of ca


for t in t_range:
    #amount_of_ca = (2/3)*np.pi*(r_step**3)*diffusion_3d(r, t, d_um2_ms, ca, t0, t0)
    amount_of_ca = (2 / 3) * np.pi * (r**2) * r_step * diffusion_3d(r, t, d_um2_ms, ca, t0, t0)
    #plt.plot(r, diffusion_3d(r, t, d_um2_ms, ca, t0, t0))
    plt.plot(r, amount_of_ca)

plt.title('Calcium diffusion')
plt.xlabel('Radius (nm)')
plt.ylabel('Concentration of calcium')
#plt.legend()
plt.show()
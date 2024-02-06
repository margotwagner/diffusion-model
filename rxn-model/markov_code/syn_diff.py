# Synaptic diffusion model
# 1/8/2020
# Margot Wagner

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats

"""
# Test
mu = 0          # mean
variance = 1
sigma = math.sqrt(variance)     # std dev
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.plot(x, stats.norm.pdf(x, mu+2, sigma))     # shift
plt.plot(x, -stats.norm.pdf(x,mu, sigma))       # opposite
plt.show()
"""

r_post = 1  # point at which concentration is zero (astrocyte location)


def p_glu(r, t, plot="on"):
    d_glu = 2.2e-4  # micron^2/microsec
    mu = 0  # mean
    variance = 2 * d_glu * t
    sigma = math.sqrt(variance)  # std dev
    p_glu = (
        stats.norm.pdf(r, mu, sigma)
        - stats.norm.pdf(r, mu + 2 * r_post, sigma)
        - stats.norm.pdf(r, mu - 2 * r_post, sigma)
    )  # zero value BC

    if plot == "on":
        plt.plot(r, p_glu)
        plt.hlines(0, -r_post - 0.5 * sigma, r_post + 0.5 * sigma)
        plt.vlines(0, -0.1, 1)
        plt.vlines(r_post, 0, 1, linestyles="dotted")
        plt.vlines(-r_post, 0, 1, linestyles="dotted")
        plt.title("Diffusion of glutamate")
        plt.xlabel("Distance from center")
        plt.ylabel("Probability")
        plt.show()

    return p_glu


r = np.linspace(-r_post, r_post, 100)
t = [500, 750, 1000, 1250]

p_array = p_glu(r, 500)

for time in t:
    p_glu(r, time)


## Bessel functions for long time (MATLAB code)
"""BesselRoots = ...  # first five roots of the Bessel function J0 and J1
[2.4048,  3.8317,  
 5.5201,  7.0156, 
 8.6537, 10.1735, 
11.7915, 13.3237,
14.9309, 16.4706 ]

BesselExtrema = ...  # first five extrema of the Bessel functions J0 and J1
[     0,  1.8412, 
 3.8317,  5.3314, 
 7.0156,  8.5363, 
10.1735, 11.7060, 
13.3237, 14.8636 ];

% Linear plot as a function of x (theta = 0 and pi)
x = -1:0.01:1;
besselx = zeros(length(x),5);
for n = 0:4
    figure(n+1);
    for i = 1:5
        besselx(:,i) = besselj(n, x*BesselRoots(i,n+1));
    end
    plot(x, besselx);
    hold on
    plot([-1,1],[0,0],'--k')
    hold off
    axis([-1.1 1.1 -1.1 1.1])
    title(['n = ', num2str(n)])
    xlabel('x (units R)')
    ylabel(['J_{', num2str(n), '}((\lambda_{', num2str(n), 'i})^{1/2}\rho)'])
    print('-depsc', ['cylindrical_x_', num2str(n), '.eps'])
end

"""

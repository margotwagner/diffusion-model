'''
Script for modeling calcium diffusion in a rectangular tube with zero flux in the x-, y-, and z-directions.

 '''

# import packages
'''
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from scipy import stats
'''
import math

# CONSTANTS
L_x = 4     # X LENGTH (um)
L_y = 0.5   # Y LENGTH (um)
L_z = 0.5   # Z LENGTH (um)
D = 220     # Diffusion coeff (um^2/sec)
x_o = 2.35  # VDCC X LOC (um)
y_o = 0.25  # VDCC Y LOC (um)
z_o = 0     # VDCC Z LOC (um)
t_o = 0     # C rel time
x = 2       # SNARE X LOC (um)
y = 0.25    # SNARE Y LOC (um)
z = 0       # SNARE Z LOC (um)
K = 3       # X eigenmodes
L = 3       # Y eigenmodes
M = 3       # Z eigenmodes

t = 0.5     # time of interest

def g_xk(x, t, x_o, t_o, k): 
    '''
    Green's component for the x-direction
    '''

    g_xk = (2/L_x) * math.cos((k*math.pi*x_o)/L_x) * math.cos((k*math.pi*x)/L_x)\
        * math.exp(-D*(k*math.pi/L_x)**2*(t - t_o))

    return g_xk

def g_yl(y, t, y_o, t_o, l):
    '''
    Green's component for the y-direction
    '''

    g_yl = (2/L_y) * math.cos((l*math.pi*y_o)/L_y) * math.cos((l*math.pi*y)/L_y)\
        * math.exp(-D*(l*math.pi/L_y)**2*(t - t_o))
        
    return g_yl

def g_zm(z, t, z_o, t_o, m):
    '''
    Green's component for the z-direction
    '''

    g_zm = (2/L_z) * math.cos((m*math.pi*z_o)/L_z) * math.cos((m*math.pi*z)/L_z)\
        * math.exp(-D*(m*math.pi/L_z)**2*(t - t_o))
        
    return g_zm

def G(x, y, z, t, x_o, y_o, z_o, t_o, K, L, M):
    '''
    Overall Green's function for diffusion in the x-, y-, and z-directions
    '''

    G = 0 # initialize

    # sum across all eigenmodes of interest
    for k in range(K):
        for l in range(L):
            for m in range(M):
                G += g_xk(x, t, x_o, t_o, k) * g_yl(y, t, y_o, t_o, l) * g_zm(z, t, z_o, t_o, m)

    return G

def u(N_ca, x, y, z, t, x_o, y_o, z_o, t_o, K, L, M):
    '''
    Resulting amount of calcium at point (x, y, z) at time t
    '''
    u = N_ca*G(x, y, z, t, x_o, y_o, z_o, t_o, K, L, M)
    
    return u



print(u(N_ca, x, y, z, t, x_o, y_o, z_o, t_o, K, L, M))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:49:51 2022

@author: yangelaxue
"""

#%% Imports.

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),os.pardir))

from utils.gradient_utils import inv_lapl_FT, gradient_FT

#%% Global variables.

# c = 299792458 #ms-1
# hbar = 1.054571817e-34 #Js
# G = 6.674e-11 # m3kg-1s-2

# # Natural units.
# E_Pl = (hbar*c**5/G)**.5
# l_Pl = (hbar*G/c**3)**.5
# Edens_Pl = E_Pl/l_Pl**3
# t_Pl = (hbar*G/c**5)**.5

#%% Functions.

def get_gravpot(Edens_vals, domain_width):
    """
    Calculates the gravitational potential field energy density field. If inputs
        are given in natural units, the potential is measured in units of c^2.

    Parameters
    ----------
    Edens_val : np.ndarray
        Energy density of a field.
    domain_width : tuple, list, np.ndarray
        The height, width, depth of the volume enclosing the oscillons.

    Returns
    -------
    potential : np.ndarray
        Gravitational potential field in natural units.
    """
    
    dxdydz = domain_width/Edens_vals.shape
    lapl_Edens = inv_lapl_FT(Edens_vals, dxdydz)
    
    potential = 4*np.pi*lapl_Edens
    
    return potential

def get_gravforce_from_pot(grav_pot,domain_width):
    """
    Calculate the gravitational vector field given gravitational potential in cartesian
        coordinates by taking the gradient of said potential field.

    Parameters
    ----------
    grav_pot : np.ndarray
        Gravitational potential field.
    domain_width : tuple, list, np.ndarray
        The height, width, depth of the volume enclosing the oscillons.

    Returns
    -------
    grav_force : list
        List of gravitational accceleration in the x, y and z directions respectively.
    """
    
    dxdydz = domain_width/grav_pot.shape
    
    return [-grad for grad in gradient_FT(grav_pot,dxdydz)]
    
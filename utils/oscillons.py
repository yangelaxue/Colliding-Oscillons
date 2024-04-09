#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions which deal with finding properties of oscillons individually and as a group.

Created on Fri Nov  4 17:54:48 2022

@author: yangelaxue
"""

#%% Imports.

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),os.pardir))

from utils.label_utils import get_labels, get_zero_mask

#%% Global variables.

# c = 299792458 #ms-1
# hbar = 1.054571817e-34 #Js
# G = 6.674e-11 # m3kg-1s-2

# # Natural units.
# E_Pl = (hbar*c**5/G)**.5
# l_Pl = (hbar*G/c**3)**.5
# Edens_Pl = E_Pl/l_Pl**3
# t_Pl = (hbar*G/c**5)**.5

#%% Functions for individual oscillons.

def get_oscillon_masses(density, domain_width, oscillons, labels=None):
    """
    Calculates the mass of a chosen oscillon. Unit choice is to be determined in use.

    Parameters
    ----------
    density : np.ndarray
        Array of energy or mass densities.
    domain_width : np.ndarray
        The height, width, depth of the domain volume.
    oscillons : np.ndarray
        The labels of oscillons.
    label : int
        Which oscillon we wish to calculate its mass.

    Returns
    -------
    mass : float
        Mass of oscillon of interest defined by oscillons.
    """

    if labels==None:
        labels = get_labels(oscillons)
    labels = labels if type(labels) is list else [labels]

    dV = np.product(domain_width/density.shape)
    masses = np.array([(density[oscillons==label]*dV).sum() for label in labels])
    
    return masses

def get_oscillon_coms(density, domain_width, oscillons, labels=None):
    """
    Calculates the center of mass of each oscillon, taking into account the wrapping
        some oscillons may exhibit given the periodicity of the simulation volume.

    Parameters
    ----------
    density : np.ndarray
        Array of energy or mass densities.
    domain_width : tuple, list, np.ndarray
        The height, width, depth of the volume enclosing the oscillons.
    oscillons : np.ndarray
        The labels of oscillons.
    labels : int, list, None, optional
        Which oscillon we wish to calculate the center of mass of. If None, calculate
        the center of mass of all oscillons. The default is None.

    Returns
    -------
    coms_idxes : list
        The nearest index of the center of mass as a tuple or list of tuples.
    coms : np.ndarray, list
        The center of mass or list of.
    """

    if labels==None:
        labels = get_labels(oscillons)
    labels = labels if type(labels) is list else [labels]
    
    shape = density.shape
    
    coms_indices = []
    coms = []
    for label in labels:
        where = np.array(np.where(oscillons==label))
        indices = np.indices(shape)
        
        com_indices = []
        
        for i, (indices_x,shape_x,where_x) in enumerate(zip(indices,shape,where)):
            
            indices_x_nomod = indices_x.copy()
            if (0 in where_x) and (shape[i]-1 in where_x):
                indices_x_nomod[indices_x_nomod<shape[i]//2] += shape[i]
    #       
            com_indices.append((density*(oscillons==label)*indices_x_nomod).sum()/(density*(oscillons==label)).sum())
        
        com_indices = np.array(com_indices)
        coms.append(com_indices*domain_width/shape)
        coms_indices.append(tuple(round(com_idx) for com_idx in com_indices))
    
    return coms_indices, coms

def get_distance_between_oscillons(osc_coms, domain_width):
    """
    Returns a matrix of distances between all oscillons and every other oscillon.

    Parameters
    ----------
    osc_coms : list, np.ndarray
        Center of mass of all oscillons which to calculate the distances between.
    domain_width : tuple, list, np.ndarray
        The height, width, depth of the volume enclosing the oscillons.

    Returns
    -------
    distances : np.ndarray
        Matrix of distances between all oscillons with every other oscillon.
    """
    
    coms_xyz = [coms_x for coms_x in np.array(osc_coms).T]
    
    coms_diff_xyz = []
    for coms_x, width in zip(coms_xyz,domain_width):
        coms_diff_x = np.tile(coms_x,(coms_x.size,1))
        coms_diff_x -= coms_x[:,np.newaxis]
        
        coms_diff_x[np.where(coms_diff_x>width/2)] -= width
        coms_diff_x[np.where(coms_diff_x<-width/2)] += width
        
        coms_diff_xyz.append(coms_diff_x)
        
    return np.sqrt(np.sum([coms_diff_x**2 for coms_diff_x in coms_diff_xyz],axis=0))

def get_escape_radius(osc_masses, rho_0):
    """
    Compares the minimum speed which objects need to be flying away from an ascillon
        to escape its gravitational orbit as a function of distance from the oscillon
        with the Hubble speed of objects away from the oscillon; i.e. v_esc=v_H.
        From this we can calculate the distance away from the oscillon where objects
        within this distance will collapse toward the oscillon and objects further
        will fly away.
        (Generally, this distance will change as a function of time for each oscillon).

    Parameters
    ----------
    osc_masses : float, list
        Total mass of an oscillon or list of masses.
    rho_0 : float
        Average mass density of the universe (or volume).

    Returns
    -------
    radius : float, np.ndarray
        As described.
    """
    
    return (3*osc_masses/(4*np.pi*rho_0))**(1/3)

def get_WignerSeitz_radius(num_density):
    """
    Radius of separation is defined but having 1 particle per spherical volume of
        said radius. i.e. r such that n*V_sph==1.

    Parameters
    ----------
    num_density : float
        Number density of particles per unit volume.

    Returns
    -------
    rad : float
        Radius of separation between particles.
    """
    return (3/(4*np.pi*num_density))**(1/3)

def get_radius(domain_width, oscillons, labels):
    """
    Get the effective radius of an oscillon assuming it is approximately spherical.

    Parameters
    ----------
    oscillons : np.ndarray
        The labels of oscillons.
    labels : int, list, None, optional
        Which oscillon we wish to calculate the center of mass of. If None, calculate
        the center of mass of all oscillons. The default is None.
    domain_width : tuple, list, np.ndarray
        The height, width, depth of the volume enclosing the oscillons.

    Returns
    -------
    radius : float, np.ndarray
        As described.
    """

    if labels==None:
        labels = get_labels(oscillons)
    labels = labels if type(labels) is list else [labels]
    
    dV = np.product(domain_width/oscillons.shape)
    Vs = [(oscillons==label).sum()*dV for label in labels]
    return np.array([(3*V/(4*np.pi))**(1/3) for V in Vs])

def get_Schwarzschild_radius(masses):
    """
    Returns the Schwarzschild radius of a mass.
    """

    return 2*masses

def get_radial_profile(domain_width, *vals_sh):
    """
    Retrieves the radial profile field. This function assumes that the object of
        interest (usually oscillon) is centered and that all spacial directions are
        of the same scale (i.e. no distortions).

    Parameters
    ----------
    domain_width : iterable
        Domain width of the vals space. Used to shift and return coordinates.
    *vals_sh : np.ndarray
        Field values.

    Returns
    -------
    r : np.ndarray
        Radial spacial coordinate, only extending to the face of the cartesian volume.
    radial_profile : np.ndarray
        Profile of vals from the center of the volume.
    """
    
    shape = vals_sh[0].shape
    
    _XYZ = np.indices(shape)
    XYZ = np.array([_X-dimension//2 for _X,dimension in zip(_XYZ,shape)])
    R = (np.sum([X**2 for X in XYZ], axis=0)**.5).astype(int)
    
    bin_count = np.bincount(R.ravel())
    vals_sh_count = tuple(np.bincount(R.ravel(), val_sh.ravel()) for val_sh in vals_sh)
    
    r = np.linspace(0,min(domain_width)/2,min(shape)//2)
    radial_profiles = tuple((val_sh_count/bin_count)[:min(shape)//2] for val_sh_count in vals_sh_count)
    
    return r, radial_profiles

def get_oscllon_numdens(oscillons, domain_width):
    """
    Calculates the number density of oscillons in a given volume.
    """
    
    N = oscillons.max()+1
    V = np.product(domain_width)
    
    return N/V
    
#%% Test

def main():
    pass

if __name__=="__main__":
    main()
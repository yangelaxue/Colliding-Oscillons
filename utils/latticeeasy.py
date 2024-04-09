#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 13:06:35 2023

@author: yangelaxue
"""

#%% Imports.

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),os.pardir))

from utils.gradient_utils import gradient_discrete
from utils.gmon import V_pr, get_A, get_B, get_r, get_s
from utils.label_utils import label_blobs

#%% Global varables.

c = 299792458. # ms-1
hbar = 1.054571817e-34 #Js
G = 6.674e-11 #mekg-1s-2

E_Pl = (hbar*c**5/G)**.5
l_Pl = (hbar*G/c**3)**.5
Edens_Pl = E_Pl/l_Pl**3
t_Pl = (hbar*G/c**5)**.5

#%% Functions for derived fields.

def Edens_T_pr(phi_pr, phip_pr, a, ap, r):
    """
    Kinetic energy density of the scalar field in program units.
    """
    return .5*phip_pr**2 - r*ap/a*phi_pr*phip_pr + .5*r**2*ap**2/a**2*phi_pr**2

def Edens_G_pr(phi_pr, a, s, dxdydz_pr):
    """
    Gradient energy density of the scalar field in program units.
    """
    grad_squared = np.array(gradient_discrete(phi_pr, dxdydz_pr, 5))**2
    return .5*a**(-2*s-2) * np.sum(grad_squared,axis=0)

def get_Edens_pr(phi_pr, phip_pr, a, ap, alpha, beta, phi_0, dxdydz_pr):

    rescale_r, rescale_s = get_r(alpha), get_s(alpha)

    return (
        Edens_T_pr(phi_pr, phip_pr, a, ap, rescale_r)
        + Edens_G_pr(phi_pr, a, rescale_s, dxdydz_pr)
        + V_pr(phi_pr, a, alpha, beta, phi_0)
        )

def get_momenta_pr(phi_pr, phip_pr, a, ap, alpha, beta, phi_0, dxdydz_pr):
    """
    Get momentum field in the x, y, z directions of data in code units.
    P^i_pr = grad_pr(phi_pr) * (phip_pr - r a'/a phi_pr)
    """
    rescale_r = get_r(alpha)
    grad_xyz = gradient_discrete(phi_pr, dxdydz_pr, 5)
    momenta_pr = -np.array([grad_x*(phip_pr-rescale_r*ap/a*phi_pr) for grad_x in grad_xyz])

    return momenta_pr

def get_field_speed_pr(momenta_pr, Edens_pr, a, alpha): #TODO I don't like this function.
    """
    Calculates the average field speed in code units.
    I assume v=sum(momenta)/sum(Edens) is a comoving velocity.
    """
    rescale_s = get_s(alpha)
    return a**(-2*rescale_s-2) * np.array([momentum_pr_x.sum()/(Edens_pr).sum() for momentum_pr_x in momenta_pr])

def get_momenta(phi_pr, phip_pr, a, ap, alpha, beta, phi_0, dxdydz_pr):
    """
    Get momentum field in the x, y, z directions of data in natural units.
    P_i = B^2/A^2 a^(s-2r) * -1/a^2 * P^i_pr
    """
    rescale_A, rescale_B, rescale_r, rescale_s = get_A(phi_0), get_B(alpha,beta,phi_0), get_r(alpha), get_s(alpha)
    grad_xyz = gradient_discrete(phi_pr, dxdydz_pr, 5)
    scale = rescale_B**2/rescale_A**2 * a**(rescale_s-2*rescale_r)
    return scale * -1/a**2 * np.array([grad_x*(phip_pr-rescale_r*ap/a*phi_pr) for grad_x in grad_xyz])

def get_field_speed(momenta, Edens, a): #TODO I don't like this function.
    """
    Calculates the average field speed in natural units.
    """

    return a * np.array([momentum_x.sum()/(Edens).sum() for momentum_x in momenta])

#%% Load data.

def load_info(output_dir,info_fname=None):
    """
    Retrieve run parameters from a file.
    """
    
    if info_fname is None:
        info_fdir = os.path.join(output_dir,'info_0.dat')
    else:
        info_fdir = os.path.join(output_dir,info_fname)
    
    with open(info_fdir, 'r') as info_f:
        for line in info_f.readlines():
            if line.startswith('alpha ='):
                alpha = float(line.split('=')[-1])
            elif line.startswith('beta ='):
                beta = float(line.split('=')[-1])
            elif line.startswith('Grid size='):
                domain_N = int(line.split('=')[-1][:-3])
                dim = int(line[-2])
            elif line.startswith('L='):
                domain_L = float(line.split('=')[-1])
            elif line.startswith('dt='):
                dt_pr = float(line.split(',')[0].split('=')[-1])
            elif line.startswith('f0='):
                phi_0 = float(line.split(',')[-1].split('=')[-1])
    
    return alpha, beta, phi_0, domain_N, domain_L, dt_pr, dim

def load_info_all(output_dir):
    """
    Retrieve run parameters from all info_{}.dat files.
    """
    
    info_fnames = 'info_{}.dat'
    N = 0
    
    while os.path.exists(os.path.join(output_dir,info_fnames.format(N))):
        _alpha, _beta, _phi_0, _domain_N, _domain_L, _dt_pr, _dim = load_info(output_dir, info_fnames.format(N))
        
        if N==0:
            alpha, beta, phi_0, domain_N, domain_L, dt_pr, dim = _alpha, _beta, _phi_0, _domain_N, _domain_L, _dt_pr, _dim
        else:
            assert alpha==_alpha, "Conflicting run parameters are stored."
            assert beta==_beta, "Conflicting run parameters are stored."
            assert phi_0==_phi_0, "Conflicting run parameters are stored."
            assert domain_N==_domain_N, "Conflicting run parameters are stored."
            assert domain_L==_domain_L, "Conflicting run parameters are stored."
            assert dt_pr==_dt_pr, "Conflicting run parameters are stored."
            assert dim==_dim, "Conflicting run parameters are stored."
    
        N += 1
    
    return alpha, beta, phi_0, domain_N, domain_L, dt_pr, dim

def load_slicetimes(output_dir,slicetimes_fname=None):
    """
    Retrieve run times from a file.
    """
    
    if slicetimes_fname is None:
        slicetimes_fdir = os.path.join(output_dir,'slicetimes_0.dat')
    else:
        slicetimes_fdir = os.path.join(output_dir,slicetimes_fname)
    
    slicetimes = np.loadtxt(slicetimes_fdir,delimiter=None)
    
    return slicetimes

def load_slicetimes_all(output_dir):
    """
    Retrieve run times from all slicetimes_{}.dat files.
    """
    
    slicetimes_fnames = 'slicetimes_{}.dat'
    slicetimes = np.empty((0))
    N = 0
    
    while os.path.exists(os.path.join(output_dir, slicetimes_fnames).format(N)):
        times = load_slicetimes(output_dir, slicetimes_fnames.format(N))
        if times.size==1:
            times = times[np.newaxis]
        slicetimes = np.concatenate((slicetimes, times))
        N += 1
    
    return slicetimes

def load_slicetime(output_dir,t_idx):
    """
    From a series of slicetimes_{}.dat files, return only the t_idx'th time slice.
    """
    
    slicetime = load_slicetimes_all(output_dir)[t_idx]
    
    return slicetime

def load_spectra(output_dir):
    """
    Seven quantities relating to spectra:
        k, number of points in each k point, w_k^2, |fk|^2 , |fk'|^2 , nk , and rho_k.
    
    See LATTICEEASY documentation.
    """

    spectra_times_fnames = os.path.join(output_dir, "spectratimes_{}.dat")
    spectra_times = np.empty((0))
    N = 0

    while os.path.exists(spectra_times_fnames.format(N)):
        times = np.loadtxt(spectra_times_fnames.format(N),delimiter=None)
        if times.size==1:
            times = times[np.newaxis]
        spectra_times = np.concatenate((spectra_times, times))
        N += 1
    
    spectra_fnames = os.path.join(output_dir, "spectra0_{}.dat")
    spectra = np.empty((0, 7))
    
    N = 0

    while os.path.exists(spectra_fnames.format(N)): 
        _spectra = np.loadtxt(spectra_fnames.format(N),delimiter=None)
        _spectra = _spectra[np.newaxis] if _spectra.ndim==1 else _spectra
        _spectra = _spectra[:,[0,1,2,3,4,5,6]]

        spectra = np.concatenate((spectra, _spectra))

        N += 1
    
    return spectra_times, spectra.reshape((spectra_times.size,-1,7))

def load_sfs(output_dir, sf_fname=None):
    """
    Retrieve scale factor times and scale factors from a file.
    """
    
    if sf_fname is None:
        sf_fdir = os.path.join(output_dir,'sf_0.dat')
    else:
        sf_fdir = os.path.join(output_dir,sf_fname)
    
    
    
    _sfs = np.loadtxt(sf_fdir,delimiter=None)
    _sfs = _sfs[np.newaxis] if _sfs.ndim==1 else _sfs
    sfs = _sfs[:,[0,1,2,3]]
    
    return sfs

def load_sfs_all(output_dir):
    """
    Retrieve scale factor times and scale factors from all sf_{}.dat files.
    """
    
    sf_fnames = 'sf_{}.dat'
    sfs = np.empty((0,4))
    N = 0
    
    while os.path.exists(os.path.join(output_dir,sf_fnames.format(N))):
        sfs = np.concatenate((sfs, load_sfs(output_dir, sf_fnames.format(N))))
        N += 1
    
    return sfs

def load_sf(output_dir, t_idx):
    """
    From a series of sf_{}.dat files, return only the t_idx'th time slice.
    """
    
    sf = load_sfs_all(output_dir)[t_idx,:]
    
    return sf

def load_a(output_dir,info_fname=None):
    """
    Retrieve run parameters from a file.
    """
    
    if info_fname is None:
        info_fdir = os.path.join(output_dir,'info_0.dat')
    else:
        info_fdir = os.path.join(output_dir,info_fname)
    
    a = None
    with open(info_fdir, 'r') as info_f:
        for line in info_f.readlines():
            if line.startswith('a='):
                a = float(line.split(',')[-1].split('=')[-1][:-1])
    if a is None:
        a = 1.
    
    return a

def load_energies(output_dir, energy_fname=None):
    """
    Retrieve average energy times and average energies from a file.
    """
    
    if energy_fname is None:
        energy_fdir = os.path.join(output_dir,'energy_0.dat')
    else:
        energy_fdir = os.path.join(output_dir,energy_fname)
    
    
    
    _energies = np.loadtxt(energy_fdir,delimiter=None)
    _energies = _energies[np.newaxis] if _energies.ndim==1 else _energies
    energies = _energies[:,[0,1,2,3]]
    
    return _energies

def load_energies_all(output_dir):
    """
    Retrieve average energy factor times and average energies from all energy_{}.dat files.
    """
    
    energy_fnames = 'energy_{}.dat'
    energies = np.empty((0,4))
    N = 0
    
    while os.path.exists(os.path.join(output_dir,energy_fnames.format(N))):
        energies = np.concatenate((energies, load_energies(output_dir, energy_fnames.format(N))))
        N += 1
    
    return energies

def load_energy(output_dir, t_idx):
    """
    From a series of energy_{}.dat files, return only the t_idx'th time slice.
    """
    
    energy = load_energies_all(output_dir)[t_idx]
    
    return energy

def load_phis(output_dir, domain_dimension, phi_fname=None):
    """
    Retrieve scale factors from a file.
    """
    
    if phi_fname is None:
        phi_fdir = os.path.join(output_dir,'slices0_0.dat')
    else:
        phi_fdir = os.path.join(output_dir,phi_fname)
    
    phis = np.loadtxt(phi_fdir,delimiter=None).reshape((-1,*domain_dimension))
    
    return phis

def load_phis_all(output_dir,domain_dimension):
    """
    Retrieve scale factors from all sf_{}.dat files.
    """
    
    phis_fnames = 'slices0_{}.dat'
    phis = np.empty((0,*domain_dimension))
    N = 0
    
    while os.path.exists(os.path.join(output_dir,phis_fnames.format(N))):
        phis = np.concatenate((phis, load_phis(output_dir, domain_dimension, phis_fnames.format(N))))
        N += 1
    
    return phis

def load_phi(output_dir, domain_dimension, t_idx):
    """
    From a series of sf_{}.dat files, return only the t_idx'th time slice.
    """
    
    phis_fnames = 'slices0_{}.dat'
    N = 0
    
    while os.path.exists(os.path.join(output_dir,phis_fnames.format(N))):
        f_len = sum(1 for line in open(os.path.join(output_dir,phis_fnames.format(N))))//(np.product(domain_dimension))
        if f_len<t_idx+1:
            t_idx -= f_len
            N += 1
            continue
        else:
            break
    
    phi = np.loadtxt(
        os.path.join(output_dir,phis_fnames.format(N)), delimiter=None, skiprows=(np.product(domain_dimension)+1)*t_idx, max_rows=(np.product(domain_dimension))
        ).reshape(*domain_dimension)
    
    return phi

def load_phips(output_dir, domain_dimension, phip_fname=None):
    """
    Retrieve scale factors from a file.
    """
    
    if phip_fname is None:
        phip_fdir = os.path.join(output_dir,'slicesp0_0.dat')
    else:
        phip_fdir = os.path.join(output_dir,phip_fname)
    
    phips = np.loadtxt(phip_fdir,delimiter=None).reshape((-1,*domain_dimension))
    
    return phips

def load_phips_all(output_dir,domain_dimension):
    """
    Retrieve scale factors from all sf_{}.dat files.
    """
    
    phips_fnames = 'slicesp0_{}.dat'
    phips = np.empty((0,*domain_dimension))
    N = 0
    
    while os.path.exists(os.path.join(output_dir,phips_fnames.format(N))):
        phips = np.concatenate((phips, load_phips(output_dir, domain_dimension, phips_fnames.format(N))))
        N += 1
    
    return phips

def load_phip(output_dir, domain_dimension, t_idx):
    """
    From a series of sf_{}.dat files, return only the t_idx'th time slice.
    """
    
    phips_fnames = 'slicesp0_{}.dat'
    N = 0
    
    while os.path.exists(os.path.join(output_dir,phips_fnames.format(N))):
        f_len = sum(1 for line in open(os.path.join(output_dir,phips_fnames.format(N))))//(np.product(domain_dimension))
        if f_len<t_idx+1:
            t_idx -= f_len
            N += 1
            continue
        else:
            break
    
    phip = np.loadtxt(
        os.path.join(output_dir,phips_fnames.format(N)), delimiter=None, skiprows=(np.product(domain_dimension)+1)*t_idx, max_rows=(np.product(domain_dimension))
        ).reshape(*domain_dimension)
    
    return phip

#%% Start class

class LATTICEEASY:

    def __init__(self, output_dir:str):

        self.output_dir = output_dir

        self.alpha, self.beta, self.phi_0, domain_N, domain_L_pr, self.dt_pr, self.dim = load_info_all(output_dir)
        self.domain_width_pr = np.array([domain_L_pr]*self.dim)
        self.domain_dimensions = (domain_N,) * self.dim

        slice_times = load_slicetimes_all(output_dir)
        sf_times, a, ap, app = load_sfs_all(output_dir).T

        if a.size==0:
            self.t = slice_times
            self._slice_tidx = np.arange(self.t.size)
            self.a = np.ones(len(self.t))*load_a(output_dir)
            self.ap = np.zeros(len(self.t))
            self.app = np.zeros(len(self.t))
        else:
            self.t, self._slice_tidx, _sf_tidx = np.intersect1d(slice_times, sf_times, return_indices=True)
            self.a = a[_sf_tidx]
            self.ap = ap[_sf_tidx]
            self.app = app[_sf_tidx]

        self.rescale_A, self.rescale_B, self.rescale_r, self.rescale_s = get_A(self.phi_0), get_B(self.alpha, self.beta, self.phi_0), get_r(self.alpha), get_s(self.alpha)

    #%% Functions relating to momenta.

    def get_momenta_pr(self, t_idx):
        """
        Get momentum field in the x, y, z directions of data in code units.
        """
        
        phi_pr, phip_pr = load_phi(self.output_dir, self.domain_dimensions, self._slice_tidx[t_idx]), load_phip(self.output_dir, self.domain_dimensions, self._slice_tidx[t_idx])

        return get_momenta_pr(phi_pr, phip_pr, self.a[t_idx], self.ap[t_idx], self.alpha, self.beta, self.phi_0, self.domain_width_pr/self.domain_dimensions)
    
    def get_field_speed_pr(self, t_idx:int): #TODO I don't like this function.
        """ Given a t_idx, return the average speed of the field in physical natural units. """

        momenta_pr = self.get_momenta_pr(t_idx)
        Edens_pr = self.get_Edens_pr(t_idx)

        return get_field_speed_pr(momenta_pr, Edens_pr, self.a[t_idx], self.alpha)

    #%% Functions relating to energy density.

    def get_Edens_scale(self,t_idx:int):
        """ Scale factor which converts energy in program units to natural units. """
        return self.rescale_B**2/self.rescale_A**2*self.a[t_idx]**(2*self.rescale_s-2*self.rescale_r)

    def get_Edens_pr(self, t_idx:int):
        """ Calculate the total energy density of the field of a single given timestep. """

        assert type(t_idx) is int, "This function only returns Edens for one time slice."
        phi_pr, phip_pr = load_phi(self.output_dir, self.domain_dimensions, self._slice_tidx[t_idx]), load_phip(self.output_dir, self.domain_dimensions, self._slice_tidx[t_idx])
        return get_Edens_pr(phi_pr, phip_pr, self.a[t_idx], self.ap[t_idx], self.alpha, self.beta, self.phi_0, self.domain_width_pr/self.domain_dimensions)

    def get_Edens_T_pr(self, t_idx:int):
        """ Calculate the kinetic energy density of the field of a single given timestep. """

        assert type(t_idx) is int, "This function only returns Edens for one time slice."
        phi_pr, phip_pr = load_phi(self.output_dir, self.domain_dimensions, self._slice_tidx[t_idx]), load_phip(self.output_dir, self.domain_dimensions, self._slice_tidx[t_idx])
        return Edens_T_pr(phi_pr, phip_pr, self.a[t_idx], self.ap[t_idx], self.rescale_r)

    def get_Edens_G_pr(self, t_idx:int):
        """ Calculate the gradient energy density of the field of a single given timestep. """

        assert type(t_idx) is int, "This function only returns Edens for one time slice."
        phi_pr = load_phi(self.output_dir, self.domain_dimensions, self._slice_tidx[t_idx])
        return Edens_G_pr(phi_pr, self.a[t_idx], self.rescale_s, self.domain_width_pr/self.domain_dimensions)
    
    def get_Edens_V_pr(self, t_idx:int):
        """ Calculate the potential energy density of the field of a single given timestep. """

        assert type(t_idx) is int, "This function only returns Edens for one time slice."
        phi_pr = load_phi(self.output_dir, self.domain_dimensions, self._slice_tidx[t_idx])
        return V_pr(phi_pr, self.a[t_idx], self.alpha, self.beta, self.phi_0)
    
    #%% Functions relating to oscillons.

    def get_oscillons(self, t_idx:int, n_means:float, min_n_cells:int, periodic:bool=True, Edens=None):
        """ Label the oscillons at a single timeslice. """
        
        if Edens is None:
            Edens = self.get_Edens_pr(self._slice_tidx[t_idx])

        return label_blobs(Edens, n_means*Edens.mean(), min_n_cells, periodic)

# %%

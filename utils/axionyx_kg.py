#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:33:41 2022

@author: yangelaxue
"""

#%% Imports.

import numpy as np

from yt.frontends.boxlib.api import AMReXDataset
from yt.data_objects.time_series import DatasetSeries

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),os.pardir))

from utils.gradient_utils import lapl_stencil
from utils.gmon import get_beta, get_A, get_B, get_r, get_s
from utils.latticeeasy import get_Edens_pr
from utils.label_utils import label_blobs

#%% Global varables.

c = 299792458. # ms-1
hbar = 1.054571817e-34 #Js
G = 6.674e-11 #mekg-1s-2

E_Pl = (hbar*c**5/G)**.5
l_Pl = (hbar*G/c**3)**.5
Edens_Pl = E_Pl/l_Pl**3
t_Pl = (hbar*G/c**5)**.5

#%% Functions specific to the YT package.

class AMReXDatasetSeries(DatasetSeries):
    """
    Load AMReX dataset as time series for YT.
    """
    _dataset_cls = AMReXDataset

def add_EdensRel(field,data):
    """
    Adds derived field Edens/<Edens>.
    """
    avg = data.ds.all_data().quantities.weighted_average_quantity(("boxlib","Edens"), weight=("index","ones"))
    return data[("boxlib","Edens")]/avg

def add_Edens(field,data):
    """
    Addes the field Edens.
    """
    pass

def add_GravPotential(field,data): #TODO: This isn't right.
    """
    Adds gravitational potential as a derived field in comoving (S.I.) coordinates.
        Divide by scale factor squared to get phyical (S.I.) coordinates.
    """
    
    alpha, beta, phi_0 = data.ds.parameters['KG.power'], get_beta(data.ds.parameters['KG.MASS']), data.ds.parameters['KG.KG0']
    rescale_B = get_B(alpha,beta,phi_0)
    
    Mdens = data["Edens"] * Edens_Pl/c**2
    domain_width = np.array(data.ds.domain_width.in_units('code_length')) * l_Pl/rescale_B
    shape = Mdens.shape
    dxdydz = tuple(width/dimension for width,dimension in zip(domain_width,shape))
    
    return 4*np.pi*G * lapl_stencil(Mdens,dxdydz)

def get_field_vals(ds, field):
    """
    Get the field values as a numpy array.
    """
    
    all_data_level_0 = ds.covering_grid(
        level=0, left_edge=[0, 0.0, 0.0], dims=tuple(ds.domain_dimensions),
        )
    
    return np.array(all_data_level_0[field])

#%% Get data

def get_runlog_data(results_dir, fname='runlog'):
    """ Get runlog time series data. """
    
    _data = []
    with open(os.path.join(results_dir,fname), 'r') as f:
        for line in f.readlines():
            if line.split()[0]=='#':
                continue
            _data.append(line.split())
    
    _data = np.array(_data)
    steps = _data[:,0].astype(int)
    
    min_step, max_step = steps.min(), steps.max()
    n_steps = max_step-min_step
    
    where_min, = np.where(steps==min_step)
    for w_min in where_min:
        if steps[w_min+n_steps]!=max_step:
            continue
        else:
            data = _data[w_min:w_min+n_steps+1]
            break
        print("No full length of time series data found.")
        return False
    
    data_t = {
        'steps' : data[:,0].astype(int),
        'time' : data[:,1],
        'a' : data[:,3],
        'ap' : data[:,4],
        'app' : data[:,5],
        'e-folds' : data[:,6],
        'phi' : data[:,7],
        'phip' : data[:,8],
        'ratio' : data[:,9],
        }
    
    return data_t

def get_rholog_data(results_dir, fname='rholog'):
    """ Get rholog time series data. """
    
    _data = []
    with open(os.path.join(results_dir,fname), 'r') as f:
        for line in f.readlines():
            if line.split()[0]=='#':
                continue
            _data.append(line.split())
    
    _data = np.array(_data)
    steps = _data[:,0].astype(int)
    
    min_step, max_step = steps.min(), steps.max()
    n_steps = max_step-min_step
    
    where_min, = np.where(steps==min_step)
    for w_min in where_min:
        if steps[w_min+n_steps]!=max_step:
            continue
        else:
            data = _data[w_min:w_min+n_steps+1]
            break
        print("No full length of time series data found.")
        return False
    
    data_t = {
        'steps' : data[:,0].astype(int),
        'time' : data[:,1],
        'rho_t' : data[:,2],
        'rho_g' : data[:,3],
        'rho_v' : data[:,4],
        'ratio' : data[:,5],
        }
    
    return data_t

def load_sf(ds):
        
    step = int(ds.basename[3:])
    runlog_dirs = [runlog_dir for runlog_dir in os.listdir(ds.directory) if runlog_dir.startswith('runlog')]
    
    for runlog_dir in runlog_dirs:
        runlog_data = get_runlog_data(ds.directory, runlog_dir)
        if step not in runlog_data['steps']:
            continue
        else:
            a = float(runlog_data['a'][runlog_data['steps']==step])
            ap = float(runlog_data['ap'][runlog_data['steps']==step])
            app = float(runlog_data['app'][runlog_data['steps']==step])
            break
    
    return a, ap, app

#%% Class.

class AxioNyx_KG: #TODO: Make sure we dont require this class to look at data.

    """
    #TODO: Adapt this so we can look at time series data, as opposed to just a single timeslice
    """
    
    def __init__(self,output_dir):
        
        self.ds = AMReXDataset(output_dir)
        self.output_dir = output_dir
        
        # Load parameter values.
        self.alpha = self.ds.parameters['KG.power']
        self.beta = get_beta(self.ds.parameters['KG.MASS'])
        self.phi_0 = self.ds.parameters['KG.KG0']
        self.mass = self.ds.parameters['KG.mass']
        self.domain_dimensions = tuple(self.ds.domain_dimensions)
        self.domain_width_pr = np.array(self.ds.domain_width.in_units('code_length'))
        
        self.rescale_A, self.rescale_B, self.rescale_r, self.rescale_s = get_A(self.phi_0), get_B(self.alpha, self.beta, self.phi_0), get_r(self.alpha), get_s(self.alpha)
        
        # Load scale factor and derivatives.
        self.a, self.ap, self.app = load_sf(self.ds)
        
        # Load and define physical quantities, all in physical units.
        # Edens = get_field_vals(self.ds, ("boxlib", "Edens")) * Edens_Pl
        # Edens[Edens<0] = 0
        # self.Edens = Edens
    
    def get_Edens_pr(self,):

        phi_pr = get_field_vals(self.ds, ("boxlib", "KGfpr"))
        phip_pr = get_field_vals(self.ds, ("boxlib", "KGfVpr"))

        Edens_pr = get_Edens_pr(phi_pr, phip_pr, self.a, self.ap, self.alpha, self.beta, self.phi_0, self.domain_width_pr/self.domain_dimensions)

        return Edens_pr

    # def phys_to_pu_pos(self, x):
    #     """ Convert physical coordinates to program coordinates. """
    #     return x * self.rescale_B/(self.a*l_Pl)
    
    # def pu_to_phys_pos(self, x):
    #     """ Convert program coordinates to physical coordinates. """
    #     return x * self.a*l_Pl/self.rescale_B
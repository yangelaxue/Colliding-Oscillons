#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Describes the generalised monodromy model, and includes all functions directly relating
to the model.

Created on Thu Nov  3 15:34:02 2022

@author: yangelaxue
"""

#%% Imports.

import numpy as np
from scipy import optimize

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),os.pardir))

#%% Global variables.

m_Pl = 1. # Plamck mass
M_Pl = m_Pl/(8*np.pi)**.5

#%% Get conversion factors.

get_beta = lambda Mass : M_Pl/Mass
get_Mass = lambda beta : M_Pl/beta
get_mass = lambda alpha, beta : (
    M_Pl**2*((2.4e-9)**.5)**2*96*np.pi**2*alpha**3*beta**(2-2*alpha)*(4*alpha*55)**(-1-alpha)
    )**.5 # Delta_R=2.4e-9, N=55

get_A = lambda phi_0 : 1/phi_0
get_B = lambda alpha, beta, phi_0 : get_mass(alpha,beta) * (phi_0/get_Mass(beta))**(-1+alpha)
get_r = lambda alpha : 3/(1+alpha)
get_s = lambda alpha : 3*(1-alpha)/(1+alpha)

#%% Define functions in physical (natural) units.

def V(phi, alpha, beta):
    """
    V(phi) = m^2M^2/(2 alpha) [ (1 + phi^2/M^2)^alpha - 1 ]
    """
    
    mass, Mass = get_mass(alpha,beta), get_Mass(beta)
    return mass**2*Mass**2/(2*alpha) * ((1+phi**2/Mass**2)**alpha-1)

def Vp(phi, alpha, beta):
    """
    V'(phi) = m^2 phi ( 1 + phi^2/M^2 )^(alpha-1)
    """
    
    mass, Mass = get_mass(alpha,beta), get_Mass(beta)
    return mass**2 * phi * (1+phi**2/Mass**2)**(alpha-1)

def Vpp(phi, alpha, beta):
    """
    V''(phi) = (
        m^2 ( 1 + phi^2/M^2 )^(alpha-1)
        + m^2 2 phi^2/M^2 (alpha-1) ( 1 + phi^2/M^2 )^(alpha-2)
        )
    """
    
    mass, Mass = get_mass(alpha,beta), get_Mass(beta)
    return (
        mass**2 * (1+phi**2/Mass**2)**(alpha-1)
        + mass**2 * 2*phi**2/Mass**2*(alpha-1) * (1+phi**2/Mass**2)**(alpha-2)
        )

def get_phi_end(alpha, beta, root_0:float=.1):
    """
    Find epsilon(phi_end)=1, epsilon = m_Pl^2/(16 pi) (V'/V)^2.
    """
    
    diff = lambda phi, alpha, beta : m_Pl**2/(16*np.pi)*Vp(phi,alpha,beta)**2-V(phi,alpha,beta)**2
    
    root_0 = .1 if alpha<.5 else .2 # TODO: Update all other code to reflect this change

    phi_end = optimize.newton(diff, root_0, args=(alpha,beta,))
    while phi_end<1e-5:
        root_0 += .01
        phi_end = optimize.newton(diff, root_0, args=(alpha,beta,))
    
    return phi_end

def get_mu_over_H_max(alpha, beta):
    """
    max(R(mu_k)/H) ~ A(alpha) beta, A(alpha) ~ 1/2 ( 1 - alpha - (1-alpha)^2/10 )
    """
    
    A = lambda alpha : .5*(1-alpha-.1*(1-alpha)**2)
    return A(alpha) * beta

#%% Define functions in program units.

def V_pr(phi_pr, a, alpha, beta, phi_0):
    """
    V_pr(phi_pr) = a^(-2s+2r)/(2 alpha) (phi_0/M)^(-2 alpha) [ (1+(phi_0 phi_pr/(a^r M))^2)^alpha - 1 ]
    """
    
    Mass, r, s = get_Mass(beta), get_r(alpha), get_s(alpha)
    coeff = phi_0/(a**r*Mass)
    return a**(-2*s+2*r) * (phi_0/Mass)**(-2*alpha) * ((1+(coeff*phi_pr)**2)**alpha-1) / (2*alpha)

def Vp_pr(phi_pr, a, alpha, beta, phi_0):
    """
    V_pr'(phi_pr) = a^(2s) phi_pr ( 1 + (phi_0 phi_pr/(a^r M))^2 )^(alpha-1)
    """
    
    Mass, r, s = get_Mass(beta), get_r(alpha), get_s(alpha)
    coeff = phi_0/(a**r*Mass)
    return a^(-2*s) * (phi_0/Mass)**(2-2*alpha) * phi_pr * (1+(coeff*phi_pr)**2)**(alpha-1)

def Vpp_pr(phi_pr, a, alpha, beta, phi_0):
    """
    V_pr''(phi_pr) = (
        (a^r M/phi_0)^(2 alpha-2) ( 1 + (phi_0 phi_pr/(a^r M))^2 )^(alpha-1)
        + (a^r M/phi_0)^(2 alpha-4) 2 phi_pr^2 (alpha-1) ( 1 + (phi_0 phi_pr/(a^r M))^2 )^(alpha-2)
        )
    """
    
    Mass, r, s = get_Mass(beta), get_r(alpha), get_s(alpha)
    coeff = phi_0/(a**r*Mass)
    return (
        a^(-2*s) * (phi_0/Mass)**(2-2*alpha) * (1+(coeff*phi_pr)**2)**(alpha-1)
        + a^(-2*s) * (phi_0/Mass)**(4-2*alpha) * 2*phi_pr**2*(alpha-1) * (1+(coeff*phi_pr)**2)**(alpha-2)
        )
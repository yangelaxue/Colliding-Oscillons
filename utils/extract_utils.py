#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions which help extract an oscillon, interpolate it and smack it on some background.
Order of function usage is:
    1) Crop a cube around a chosen oscillon.
    2) Interpolate it.
    3) Crops a sphere around the oscillon and places it in some background.
    4) Smooth the edges of the sphere.

Created on Mon Jan 16 10:00:43 2023

@author: yangelaxue
"""

#%% Imports

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

import sys, os
sys.path.insert(0, os.path.join(os.getcwd(),os.pardir))

from utils.oscillons import get_oscillon_coms
from utils.volume_utils import shift_volume
from utils.label_utils import find_neighbours

#%%

def get_crop_params(Edens, oscillons, label, pad:int=None):
    """
    Given a chosen oscillon, this function will provide the center of mass index
        of the oscillon and with radius encompassing it (and some padding).

    Parameters
    ----------
    Edens : np.ndarray
        The energy density field of the oscillons.
    oscillons : np.ndarray
        The labels of oscillons.
    labels : int
        Which oscillon we wish to find the center and span of.
    pad : int, tuple, None
        How many pixels away from the perimeter of the chosen oscillon do we want
            to crop.

    Returns
    -------
    center_idxes : tuple
        Center of mass indices of the oscillon.
    radius : int
        Radius which encompasses the oscillon in index units.
    """

    assert type(label) is int, "You can only find parameters for one oscillon."
    assert Edens.shape==oscillons.shape, "All fields must be of the same shape"
    
    shape = Edens.shape

    pad = max(shape)//16 if pad is None else pad

    # Get center indices.
    center_idxes, = get_oscillon_coms(Edens, shape, oscillons, labels=label)[0]

    # Get radius
    diameter = ()
    where = np.where(oscillons==label)
    for where_x, shape_x in zip(where,shape):
        idxes_x = np.unique(where_x)
        if 0 in idxes_x and shape_x-1 in idxes_x:
            idxes_x[idxes_x<shape_x//2] += shape_x
        diameter += (idxes_x.max()-idxes_x.min(),)
    
    radius = max(diameter)//2 + pad

    return center_idxes, radius

def crop(Edens, oscillons, label, pad:int=None, *vals):
    """
    Crops field values to a cube around a desired center.

    Parameters
    ----------
    Edens : np.ndarray
        The energy density field of the oscillons.
    oscillons : np.ndarray
        The labels of oscillons.
    labels : int
        Which oscillon we wish to find the center and span of.
    pad : int, tuple, None
        How many pixels away from the perimeter of the chosen oscillon do we want
            to crop.
    *vals : np.ndarray
        Field values which are to be cropped.

    Returns
    -------
    *vals_sh_cr : np.ndarray
        Cropped field values.
    """

    shape = Edens.shape
    assert Edens.shape==oscillons.shape, "All fields must be of the same shape"
    for val in vals:
        assert val.shape==shape, "All fields must be of the same shape"
    
    center_idxes, radius = get_crop_params(Edens, oscillons, label, pad)

    slices = tuple((slice(shape_x//2-radius,shape_x//2+radius) for shape_x in shape))
    vals_sh = shift_volume(center_idxes,*vals)
    vals_sh_cr = tuple(val_sh[slices] for val_sh in vals_sh)
    
    return vals_sh_cr

# def expand_mask(mask, radius, periodic:bool):

#     shape = mask.shape
#     new_mask = mask.copy()

#     for i in range(shape[0]):
#         if mask[i]==0: continue
#         for j in range(shape[1]):
#             if mask[i,j]==0: continue
#             for k in range(shape[2]):
#                 if mask[i,j,k]==0: continue
#                 if mask[find_neighbours((i,j,k), shape, periodic)]
#                 pass

    





def interpolate(shape_i:tuple, *vals):
    """
    Given a tuple of field values *vals, interpolate points inside the grid to a
        desired shape_i.

    Parameters
    ----------
    shape_i : tuple
        Shape of interpolated grid points.
    *vals : np.ndarray
        Field values which we wish to interpolate.

    Returns
    -------
    *fs_i : np.ndarray
        Interpolated values of *vals.
    """
    
    shape = vals[0].shape
    for val in vals:
        assert val.shape==shape, "All fields must be of the same shape"
    ndim = vals[0].ndim
    
    xyz = tuple(np.linspace(0,1,shape_x) for shape_x in shape)
    interpolators = tuple(RegularGridInterpolator(xyz, val, method='linear') for val in vals)
    
    xyz_i = (np.linspace(0,1,shape_i_x) for shape_i_x in shape_i)
    XYZ_i = np.array(np.meshgrid(*xyz_i, indexing='ij'))
    XYZ_i_reshape = XYZ_i.transpose(tuple((dim+1)%(ndim+1) for dim in range(ndim+1)))
    all_points = XYZ_i_reshape.flatten().reshape(np.prod(shape_i),ndim)
    
    fs_i = tuple(interpolator(all_points).reshape(shape_i) for interpolator in interpolators)
    
    return fs_i

def add_field_to_background(bg, *vals):
    """
    Crops the field as a sphere and places it in the center
        of some background.

    Parameters
    ----------
    bg : np.ndarray
        Background which to place the field values.
    *vals : np.ndarray
        Field values which are to be placed on a background.
    
    Returns
    -------
    *vals_bg : np.ndarray
        Background with the oscillon.
    """

    shape = vals[0].shape
    for val in vals:
        assert val.shape==shape, "All fields must be of the same shape"
    n_vals = len(vals)
    shape_bg = bg.shape

    radius = np.max(shape)//2

    XYZ_bg = np.indices(shape_bg)
    XYZ_bg = [X_bg - shape_bg_x//2 for X_bg,shape_bg_x in zip(XYZ_bg,shape_bg)]
    R_bg = np.sqrt(np.sum([X_bg**2 for X_bg in XYZ_bg],axis=0))

    XYZ = np.indices(shape)
    XYZ = [X - shape_x//2 for X,shape_x in zip(XYZ,shape)]
    R = np.sqrt(np.sum([X**2 for X in XYZ],axis=0))

    mask_idxes = np.where(R<radius)
    mask_bg_idxes = np.where(R_bg<radius)
    
    vals_bg = ()
    for i, val in enumerate(vals):
        val_bg = bg.copy()
        val_bg[mask_bg_idxes] = val[mask_idxes]
        vals_bg += (val_bg,)

    return vals_bg

def smooth_edge(radius, slope=None, sigma=None, *vals):
    """
    Tapers the opacity of a field at a given radius in index units from
        the center. Then blurs the fields from where the taper begins.

    Parameters
    ----------
    radius : int, list
        Radius of the taper in index units.
    slope : float, None
        How smoothly should the taper be.
    sigma : float, None
        The amount of blur to be applied.
    *vals : np.ndarray
        Field values to be smoothed.
    
    Returns
    -------
    *vals_tapered_blur : np.ndarray
        Field values with a smoothed edge.
    """

    shape = vals[0].shape
    for val in vals:
        assert val.shape==shape, "All fields must be of the same shape"
    n_vals = len(vals)
    
    if type(radius) is int or type(radius) is float:
        radius = [int(radius)] * n_vals
    if slope is None:
        slope = [1.] * n_vals
    elif type(slope) is int or type(slope) is float:
        slope = [slope] * n_vals
    if sigma is None:
        sigma = [4.] * n_vals
    elif type(sigma) is int or type(sigma) is float:
        sigma = [sigma] * n_vals
    assert len(radius)==len(slope)==len(sigma)==n_vals
    
    XYZ = np.indices(shape)
    XYZ = [X - shape_x//2 for X,shape_x in zip(XYZ,shape)]
    R = np.sqrt(np.sum([X**2 for X in XYZ],axis=0))
    
    sigmoid = lambda R, radius, slope : 1/(1+np.exp(slope*(R-radius)))
    vals_taper = tuple(val*sigmoid(R, rad, slop) for val, rad, slop in zip(vals, radius, slope))
    
    masks = [np.zeros(shape) for i in range(n_vals)]
    for i, (rad, sig) in enumerate(zip(radius, sigma)):
        masks[i][R>(rad)] = 1
        masks[i] = gaussian_filter(masks[i], sig)

    vals_blurred = tuple(gaussian_filter(val_taper, sig) for sig, val_taper in zip(sigma, vals_taper))
    
    vals_tapered_blur = tuple(val_taper*(1-mask) + val_blurred*mask for mask, val_taper, val_blurred in zip(masks, vals_taper, vals_blurred))
    
    return vals_tapered_blur

def extract_oscillon(Edens, oscillons, label, shape_i, bg, *vals, **kwargs):
    """
    Extracts single oscillons.
    """
    
    kwargs["pad"] = kwargs["pad"] if "pad" in kwargs else None
    kwargs = {
        "slope" : .1,
        "sigma" : 4.,
        **kwargs
    }

    vals_cr = crop(Edens, oscillons, label, kwargs["pad"], *vals)
    vals_cr_i = interpolate(shape_i, *vals_cr)
    vals_cr_i_bg = add_field_to_background(bg, *vals_cr_i)
    vals_cr_i_bg_sm = smooth_edge(kwargs["smooth_radius"], kwargs["slope"], kwargs["sigma"], *vals_cr_i_bg)
    
    # Find width of box
    width_frac = np.array([shape_cr/shape for shape_cr,shape in zip(vals_cr[0].shape,vals[0].shape)])
    width_frac = width_frac*bg.shape/shape_i

    return vals_cr_i_bg_sm, width_frac
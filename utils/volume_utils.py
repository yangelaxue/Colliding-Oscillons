"""
Functions used to shift volumes of field values assuming periodic boundary conditions.
"""

#%% Imports.
import numpy as np

#%% Start functions.

def shift_volume(toshift_idxes, *vals):
    """
    Redefines a volume of data to center a chosen point. It does this by cutting
        the data at a point determined by the desired center and glueing the opposite
        faces together, i.e. making use of the periodic spacial boundary conditions.

    Parameters
    ----------
    toshift_idxes : tuple
        Indices of the point which is to be moved to the center of the volume.
    *vals : np.ndarray
        Array of data which is to be shifted and glued.

    Returns
    -------
    vals_sh : np.ndarray
        Shifted values.
    """
    
    shape = vals[0].shape
    center_idxes = tuple(shp//2 for shp in shape)

    # Set up shifted values
    vals_sh = [val.copy() for val in vals]

    for i, (center_idx, toshift_idx) in enumerate(zip(center_idxes, toshift_idxes)):
        
        # Set up slices.
        slices_l = [slice(0,shp) for shp in shape]
        slices_r = [slice(0,shp) for shp in shape]
        
        shift = (toshift_idx-center_idx)%shape[i]
        slices_l[i] = slice(0,shift)
        slices_r[i] = slice(shift,None)
        
        # Change vals.
        for j, val_sh in enumerate(vals_sh):
            _val_sh_copy = val_sh.copy()
            vals_sh[j] = np.concatenate((_val_sh_copy,_val_sh_copy[tuple(slices_l)]),axis=i)
            vals_sh[j] = vals_sh[j][tuple(slices_r)]
            
    return vals_sh

def shift_coordinates(toshifts_idxes, *xyz):
    """
    Shifts 1D coordinates to a new chosen center.

    Parameters
    ----------
    toshifts : tuple, list, iterable
        The coordinate point which will be shifted to the center.
    *xyz : np.ndarray
        Numpy 1D arrays of cooridiates.

    Returns
    -------
    xyz_sh : list
        List of newly shifted coordinates.
    """
    
    if type(toshifts_idxes) is not list:
        toshifts_idxes = list(toshifts_idxes)
    
    toshifts = []
    for idx, x in zip(toshifts_idxes,xyz):
        toshifts.append(x[idx])
    
    center = tuple(len(x)//2 for x in xyz)
    center_coord = [x[cen] for x,cen in zip(xyz,center)]
    
    # Set up shifted values
    xyz_sh = []
    
    for i, (toshift,cen_coord) in enumerate(zip(toshifts,center_coord)):
        
        _x_sh = xyz[i].copy()
        _x_sh += toshift-cen_coord
        
        xyz_sh.append(_x_sh)
        
    return xyz_sh
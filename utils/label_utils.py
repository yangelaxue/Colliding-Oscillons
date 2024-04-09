    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deals with all of the labelling of oscillons.

Created on Fri Nov  4 16:39:34 2022

@author: yangelaxue
"""

#%% Imports.

import numpy as np
from tqdm import tqdm

#%% Functions.

def find_neighbours(center:tuple, shape:tuple, periodic:bool):
    """
    Given center indices, find all directly adjacent cells, neighbours, taking
        into account of the periodicity of the data space from which shape is derived
        if wanted.

    Parameters
    ----------
    center : iterable
        Indices of the point to find the neighbours for.
    shape : iterable
        Shape which encloses the center.

    Returns
    -------
    neighbours_idx : tuple
        Indices of neighbours which can be used to select all neighbours of a center.
        Same formatting structure as numpy.where outputs.
    """
    
    neighbours = []
    
    for i,dimension in enumerate(shape):
        
        if periodic:
            nb = list(center)
            nb[i] = (nb[i]-1)%dimension
            neighbours.append(nb)
            
            nb = list(center)
            nb[i] = (nb[i]+1)%dimension
            neighbours.append(nb)
        
        else:
            nb = list(center)
            nb[i] = (nb[i]-1)
            if nb[i]>-1:
                neighbours.append(nb)
            
            nb = list(center)
            nb[i] = (nb[i]+1)
            if nb[i]<dimension:
                neighbours.append(nb)
    
    return tuple(np.array(neighbours).T)

def skewer_3D(f, threshold:float, periodic:bool):
    """  
    Given a 3D field values f, select all points greater than or equal to a threshold
        value and give a preliminary label. Attempts to label points and its neighbours
        the same label with labels starting at 0. Such points and neighbouring points
        are called 'blobs'. Function "remove_redundant_labels" and "relabel_blobs"
        is necessary to label points and neighbours distictly and uniquely.
        TODO: No such function exists for any ndim ds.
    
    Parameters
    ----------
    field_vals : np.ndarray
        Dataset which contains some localised datapoints greater than others by some factor.
    threshold : float
        Value which points in field_vals must equal or exceed to be labeled part of a blob.

    Returns
    -------
    blobs : np.ndarray
        Of same shape as field_vals, it contains -1 everywhere except where there are
        blobs in field_vals. Only preliminary labels are given.
    """
    
    assert f.ndim==3, "This function only skewers 3D volumes."
    shape = f.shape
    
    blobs = -np.ones_like(f,dtype=int)
    count = -1
    
    # loop over all dimensions, skipping entire indices whenever possible.
    for i in (tqdm_bar := tqdm(range(shape[0]))):
        tqdm_bar.set_description("Assigning preliminary labels")
        if (f[i]<threshold).all():
            continue
        for j in range(shape[1]):
            if (f[i,j]<threshold).all():
                continue
            for k in range(shape[2]):
                center = (i,j,k)
                neighbours_idxes = find_neighbours(center, shape, periodic)
                
                if f[center]<threshold:
                    continue
                elif (blobs[neighbours_idxes]==-1).all():
                    count += 1
                    blobs[center] = count
                else:
                    blobs[center] = min(blobs[neighbours_idxes][blobs[neighbours_idxes]>-1])
                    blobs[neighbours_idxes][blobs[neighbours_idxes]>blobs[center]] = blobs[center]

                threshold_neighbours = tuple(np.array(neighbours_idxes).T[np.where(f[neighbours_idxes]>threshold)].T)
                blobs[threshold_neighbours] = blobs[center]
    
    return blobs

def remove_redundant_labels(blobs,periodic:bool): #TODO: I believe there is a bug here.
    """
    Given an array of poorly labelled blobs, it goes through the blobs and labels
        each blob a unique and distict label.

    Parameters
    ----------
    blobs : np.ndarray
        Blobs and their labels.

    Returns
    -------
    blobs : np.ndarray
        Blobs are now unique and distinct.
    """
    
    blobs = blobs.copy()
    labels = get_labels(blobs)
    
    for label in (tqdm_bar := tqdm(labels)):
        tqdm_bar.set_description("Removing redundant labels")
        if label not in blobs:
            continue
        
        label_neighbours = []
        blob_idxes = np.array(np.where(blobs==label)).T
        
        for blob_idx in blob_idxes:
            blob_idx = tuple(blob_idx)
            neighbours_idxes = find_neighbours(blob_idx, blobs.shape, periodic)
            _label_neighbours = blobs[neighbours_idxes][blobs[neighbours_idxes]>-1].tolist()
            label_neighbours += _label_neighbours
        
        label_neighbours += [label]
        label_neighbours = np.unique(label_neighbours).tolist()
        min_label = np.min([label] + label_neighbours)
        for label_neighbour in label_neighbours:
            blobs[np.where(blobs==label_neighbour)] = min_label
        
    return blobs

def relabel_blobs(blobs):
    """
    Given an array of blobs and labels, it relabels all blobs in ascending order.

    Parameters
    ----------
    blobs : np.ndarray
        Blobs and their labels.

    Returns
    -------
    blobs : np.ndarray
        Blobs and their labels.
    """
    
    blobs = blobs.copy()

    labels = get_labels(blobs)
    relabel = 0
    
    for label in (tqdm_bar := tqdm(labels)):
        tqdm_bar.set_description("Relabeling blobs in ascending order")
        
        blobs[blobs==label] = relabel
        relabel += 1
    
    return blobs

def sieve(blobs, min_n_cells:int=1,):
    """
    Filter out blobs which are composed of fewer than n_cells cells large.

    Parameters
    ----------
    blobs : np.ndarray
        Blobs and their labels. Should have redundant labels removed.
    min_n_cells : int, optional
        Minimum number of cells which composes a blob. The default is 1.

    Returns
    -------
    blobs : np.ndarray
        Composed of large blobs only.
    """
    
    assert min_n_cells>=1, "Size of a blob must be greater than or equal to 1 cell."
    if min_n_cells==1:
        return blobs
    
    blobs = blobs.copy()
    labels = get_labels(blobs)
    
    for label in (tqdm_bar := tqdm(labels)):
        tqdm_bar.set_description(f"Sifting out small blobs fewer than {min_n_cells} cells")
        if (blobs==label).sum()<min_n_cells:
            blobs[blobs==label] = -1
            
    return blobs

def label_blobs(f, threshold:float, min_n_cells:int, periodic:bool):
    """
    Runs all functions necessary to produce unique and distintly labelled blobs
        in ascending order from a dataset and a threshold.
    
    Parameters
    ----------
    field_vals : np.ndarray
        Dataset which contains some localised datapoints greater than others by some factor.
    threshold : float
        Value which points in ds must equal or exceed to be labeled part of a blob.
    min_n_cells : int, optional
        Minimum number of cells which composes a blob. The default is 1.
        
    Returns
    -------
    blobs : np.ndarray
        All blobs labelled are unique and distict and labelled in ascending order.
    """
    
    blobs = skewer_3D(f,threshold, periodic)
    blobs = remove_redundant_labels(blobs, periodic)
    
    if min_n_cells!=1:
        blobs = sieve(blobs, min_n_cells)
    
    blobs = relabel_blobs(blobs)
    
    return blobs

#%% Functions which mask blobs.

def get_labels(blobs):
    """ Return list of labels of the blobs given. """
    return np.delete(np.unique(blobs),np.where(np.unique(blobs)==-1)).tolist()

def get_zero_mask(blobs,labels=None):
    """
    Create and return a mask which is all zeros except where labels are in blobs array.
    
    Parameters
    ----------
    blobs : np.ndarray
        All blobs labelled are unique and distict and labelled in ascending order.
    labels : None, int, list, optional
        Blob labels which not to mask. If None, all labels are not masked. The default is None.

    Returns
    -------
    zero_mask : np.ndarray, list
        Array of zeros except where labels are.
    """
    
    # Streamline if labels==None.
    if labels==None:
        zero_masks = np.ones(blobs.shape)
        zero_masks[blobs==-1] = 0
        return zero_masks
    
    if type(labels)==int:
        labels = [labels]
    zero_masks = []
    for i, label in enumerate(labels):
        mask = np.zeros_like(blobs)
        mask[blobs==label] = 1
        zero_masks.append(mask)
    
    zero_masks = np.array(zero_masks)
    
    return zero_masks

def get_nan_mask(blobs,labels=None):
    """
    Create and return a mask which is all nans except where labels are in blobs array.
    
    Parameters
    ----------
    blobs : np.ndarray
        All blobs labelled are unique and distict and labelled in ascending order.
    labels : None, int, list, optional
        Blob labels which not to mask. If None, all labels are not masked. The default is None.

    Returns
    -------
    nan_mask : list
        Array of nans except where labels are.
    """
    
    zero_masks = get_zero_mask(blobs,labels)
    if labels==None:
        nan_mask = zero_masks.copy()
        nan_mask[nan_mask==0] = np.nan
        return nan_mask
    
    if type(labels)==int:
        labels = [labels]
    nan_masks = zero_masks.copy()
    nan_masks.dtype='float64'
    nan_masks[zero_masks==0] = np.nan
    
    return nan_masks
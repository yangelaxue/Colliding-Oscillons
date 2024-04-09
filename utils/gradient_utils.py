#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 21:50:11 2022

@author: yangelaxue
"""

#%% Imports.

import numpy as np
from scipy.fftpack import fftn, ifftn

#%% Define functions.

def lapl_stencil(field, dxdydz):
    
    f = field.copy()
    
    # Assume periodic boundary conditions.
    for axis, dimension in enumerate(f.shape):
        
        f_slice, l_slice, na_slice = (
            [slice(0,d) for d in f.shape],
            [slice(0,d) for d in f.shape],
            [slice(0,d) for d in f.shape],)
        
        f_slice[axis] = slice(0,1)
        l_slice[axis] = slice(-1,dimension)
        na_slice[axis] = np.newaxis
        
        f = np.concatenate((f[tuple(l_slice)], f, f[tuple(f_slice)]), axis=axis)
    
    # Calculate Laplacian using a standard stencil.
    lapl = 0.
    for axis, (dx, dimension) in enumerate(zip(dxdydz,f.shape)):
        f_slices, l_slices, m_slices = (
            [slice(1,-1) for d in f.shape],
            [slice(1,-1) for d in f.shape],
            [slice(1,-1) for d in f.shape],)
        
        f_slices[axis] = slice(0,-2)
        l_slices[axis] = slice(2,dimension)
        
        lapl += (f[tuple(f_slices)] + f[tuple(l_slices)] - 2*f[tuple(m_slices)]) / dx**2
        
    return lapl

def gradient_FT(field, dxdydz):
    
    f = field.copy()
    
    kxkykz = 2*np.pi*np.array([np.fft.fftfreq(shape, dx) for (shape,dx) in zip(f.shape,dxdydz)])
    KxKyKz = np.meshgrid(*kxkykz,indexing='ij')
    
    gradient = [ifftn(fftn(f)*Kx*1j).real for Kx in KxKyKz]
    
    return gradient

def gradient_discrete(field, dxdydz, stencil:int=3):
    """
    Calculates the gradient of a field in any dimensions using a 3, 5, 7 or 9 point
        stencil. This function assumes periodic boundary conditions and even spacing
        between points.

    Parameters
    ----------
    field : np.ndarray
        Field values in arbitrary dimentions.
    dxdydz : iterable
        Step size between different values of field, one iterable for each dimension.
    stencil : int, optional
        Type of stencil used to compute the gradient. The default is 3.

    Raises
    ------
    ValueError
        Only accepts 3, 5, 7 or 9 point stencils.

    Returns
    -------
    grad : list
        List of gradients in the different directions of field.
    """
    
    f = field.copy()
    
    for axis, dimension in enumerate(f.shape):
        
        f_slice, l_slice = (
            [slice(0,dim) for dim in f.shape],
            [slice(0,dim) for dim in f.shape],)
        
        f_slice[axis] = slice(0,stencil//2)
        l_slice[axis] = slice(dimension-stencil//2,dimension)
    
        f = np.concatenate((f[tuple(l_slice)], f, f[tuple(f_slice)]), axis=axis)
        
    grad = []
    for axis, (dx,dimension) in enumerate(zip(dxdydz,f.shape)):
        
        if stencil==3:
            f_1, l_1 = (
                [slice(1,-1) for d in f.shape],
                [slice(1,-1) for d in f.shape],
            )
            
            f_1[axis] = slice(0,-2)
            l_1[axis] = slice(2,dimension)
            
            grad.append((f[tuple(l_1)]-f[tuple(f_1)])/(2*dx))
            
        elif stencil==5:
            
            f_1, f_2, l_1, l_2 = (
                [slice(2,-2) for d in f.shape],
                [slice(2,-2) for d in f.shape],
                [slice(2,-2) for d in f.shape],
                [slice(2,-2) for d in f.shape],
            )
            
            f_1[axis] = slice(1,-3)
            f_2[axis] = slice(0,-4)
            l_1[axis] = slice(3,dimension-1)
            l_2[axis] = slice(4,dimension)
            
            grad.append(
                (
                    (f[tuple(l_1)]-f[tuple(f_1)])*2/3
                    + (-f[tuple(l_2)]+f[tuple(f_2)])*1/12
                )/dx
            )
            
        elif stencil==7:
            
            f_1, f_2, f_3, l_1, l_2, l_3 = (
                [slice(3,-3) for d in f.shape],
                [slice(3,-3) for d in f.shape],
                [slice(3,-3) for d in f.shape],
                [slice(3,-3) for d in f.shape],
                [slice(3,-3) for d in f.shape],
                [slice(3,-3) for d in f.shape],
            )
            
            f_1[axis] = slice(2,-4)
            f_2[axis] = slice(1,-5)
            f_3[axis] = slice(0,-6)
            l_1[axis] = slice(4,dimension-2)
            l_2[axis] = slice(5,dimension-1)
            l_3[axis] = slice(6,dimension)
            
            grad.append(
                (
                    (f[tuple(l_1)]-f[tuple(f_1)])*3/4
                    + (-f[tuple(l_2)]+f[tuple(f_2)])*3/20
                    + (f[tuple(l_3)]-f[tuple(f_3)])*1/60
                )/dx
            )
            
        elif stencil==9:
            
            f_1, f_2, f_3, f_4, l_1, l_2, l_3, l_4 = (
                [slice(4,-4) for d in f.shape],
                [slice(4,-4) for d in f.shape],
                [slice(4,-4) for d in f.shape],
                [slice(4,-4) for d in f.shape],
                [slice(4,-4) for d in f.shape],
                [slice(4,-4) for d in f.shape],
                [slice(4,-4) for d in f.shape],
                [slice(4,-4) for d in f.shape],
            )
            
            f_1[axis] = slice(3,-5)
            f_2[axis] = slice(2,-6)
            f_3[axis] = slice(1,-7)
            f_4[axis] = slice(0,-8)
            l_1[axis] = slice(5,dimension-3)
            l_2[axis] = slice(6,dimension-2)
            l_3[axis] = slice(7,dimension-1)
            l_4[axis] = slice(8,dimension)
            
            grad.append(
                (
                    (f[tuple(l_1)]-f[tuple(f_1)])*4/5
                    + (-f[tuple(l_2)]+f[tuple(f_2)])*1/5
                    + (f[tuple(l_3)]-f[tuple(f_3)])*4/105
                    + (-f[tuple(l_3)]+f[tuple(f_3)])*1/280
                )/dx
            )
            
        else:
            raise ValueError("Can only compute 3, 5, 7 or 9 point stencils.")
        
    return grad

def inv_lapl_FT(field, dxdydz):
    
    # Get frequency space values.
    kxkykz = 2*np.pi*np.array([np.fft.fftfreq(dimension, dx) for (dimension,dx) in zip(field.shape, dxdydz)])
    kxkykz[:,0] = kxkykz[:,1]/1000
    KxKyKz = np.meshgrid(*kxkykz, indexing='ij')
    K_squared = np.sum([Kx**2 for Kx in KxKyKz], axis=0)
    
    # Calculate Laplacian.
    field_k = fftn(field)
    lapl = -ifftn(field_k/K_squared).real
    
    return lapl

#%% test

def main():
    
    field = np.zeros((16,16,16))
    field[0,:,:] = 1
    field[-1:,:,:] = -1
    
    # print(field[0,:,:])
    
    dxdydz = (1,1,1)
    
    f = lapl_stencil(field, dxdydz)
    # print(f[0,:,:])
    # print(f[:,-1,:])

if __name__=="__main__":
    
    main()
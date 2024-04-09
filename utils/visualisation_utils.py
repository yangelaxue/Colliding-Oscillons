#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shortcut functions used to visualise fields.

Created on Wed Sep 21 23:18:40 2022

@author: yange
"""

#%% Imports.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)
rc('image', cmap='gist_heat_r')
rc('axes', **{'titlesize': 12})
    
import yt

#%% 3D scatter plots.

def get_opacity(field_vals):
    """ Normalise all data to fit between 0 and 1. """
    return (field_vals-field_vals.min())/(field_vals.max()-field_vals.min())

def scatterfield_3D(field_vals,XYZ,scale_opacity:float=1.):
    
    opacity = get_opacity(np.abs(field_vals))
    
    fig = plt.figure(figsize=(6,6),tight_layout=True)
    ax = fig.add_subplot(111,projection='3d')
    
    scatter = ax.scatter(*XYZ, alpha=opacity*scale_opacity, c=field_vals)
    
    return fig, ax, scatter

def annotate_oscillons(ax,osc_labels,osc_coms):
    
    for label, com in zip(osc_labels,osc_coms):
        ax.scatter(*com, color='b', marker='x') 
        ax.text(*com, f'{label}', size=12, zorder=int(1e10), color='k')

#%% Render 3D scenes with yt

def linramp(vals,minval,maxval):
    """
    Define a linear transfer function to visualise densities.
    """
    return (vals-vals.min())/(vals.max()-vals.min())
    
def create_scene(ds, field, save:bool, save_dir:str, **scene_kwargs):
    """
    Create a volume rendering of a field.
    
    Parameters
    ----------
    ds : yt.frontends.boxlib.data_structures.AMReXDataset
        Dataset loaded onto yt.
    field : tuple
        field is of the form ("<data structure>","<field name>").
    save : bool
        Save image of rendering if true, else not. The image will be named with reference
        to d.basename of the form 'plt?????'.
    save_dir : str
        Directory which the rendering is to be saved to.
    **scene_kwargs
        Parameters to change the appearence of the rendering.

    Returns
    -------
    source : yt.visualization.volume_rendering.render_source.KDTreeVolumeSource
        Source object which can manipulate the renderings.
    """
    
    scene_kwargs = {
        'zoom' : .9,
        'resolution' : 500,
        'log' : False,
        'cmap' : 'arbre',
        'bounds' : None,
        'tf_layers' : 5,
        'grey_opacity' : False,
        'sigma_clip' : 1,
        'tf_scale_func' : None,
        **scene_kwargs
        }
    if scene_kwargs['bounds'] == None:
        bounds = ds.all_data().quantities.extrema(field[1])
        scene_kwargs.update({'bounds' : bounds})
    
    sc = yt.create_scene(ds, field)
    
    source = sc[0]
    source.set_field(field)
    source.set_log(scene_kwargs['log'])
    
    tf = yt.ColorTransferFunction(scene_kwargs['bounds'])
    if type(scene_kwargs['tf_scale_func'])==type(linramp):
        tf.map_to_colormap(*scene_kwargs['bounds'],colormap=scene_kwargs['cmap'],scale_func=scene_kwargs['tf_scale_func'])
    else:
        tf.add_layers(scene_kwargs['tf_layers'], colormap=scene_kwargs['cmap'])
    
    source.tfh.tf = tf
    source.tfh.bounds = scene_kwargs['bounds']
    source.tfh.grey_opacity = scene_kwargs['grey_opacity']
    source.tfh.set_log(scene_kwargs['log'])
    
    if save:
        source.tfh.plot(f"{save_dir}/tf_{field[1]}_{ds.basename}.png",profile_field=field)
        sc.save(f"{save_dir}/Scene_{field[1]}_{ds.basename}.png",sigma_clip=scene_kwargs['sigma_clip'])
    
    return source

def create_scenes(ds_ts, field, save:bool, save_dir:str, scene_kwargs:dict={}):
    """
    Creates multiple renderings in succession, ensuring that they share the same bounds.

    Parameters
    ----------
    See 'create_scenes' for information about parameters.

    Returns
    -------
    sources : list
        List of source objects which can manipulate the renderings.
    """
    
    extrema = np.array([d.all_data().quantities.extrema(field[1]) for d in ds_ts])
    bounds = np.array(extrema)[:,0].mean(), np.array(extrema)[:,1].mean()
    
    scene_kwargs.update({'bounds' : bounds})
    
    sources = []
    for i, ds in enumerate(ds_ts):
        sc = create_scene(ds, field, save, save_dir, **scene_kwargs)
        sources.append(sc)
        
    return sources
   
#%% Plot a grid of data for YT.

# TODO: This shit sucks!!!!!

# def plot_ytPlot(ds, fields, ytPlot_func, **plot_kwargs):
#     """
#     Produce a yt plot for a single dataset. Optimized for yt.SlicePlot.

#     Parameters
#     ----------
#     d : yt.frontends.boxlib.data_structures.AMReXDataset
#         Dataset loaded onto yt.
#     fields : tuple
#         fields is of the form ("<data structure>","<field name>").
#     ytPlot_func : function
#         ytPlot_func==yt.SlicePlot or ytPlot_func==yt.ProjectionPlot.
#     **plot_kwargs
#         Parameters to change the appearence of the plot or manipulation of data.

#     Returns
#     -------
#     ytplot : yt.visualization.plot_window.AxisAlignedSlicePlot or yt.visualization.plot_window.ProjectionPlot
#         Plot object which can be manipulated.
#     """
    
#     assert 'axis' or 'normal' in plot_kwargs.keys()
#     if 'normal' not in plot_kwargs.keys():
#         plot_kwargs.update({'normal' : plot_kwargs['axis']})
    
#     plot_kwargs = {
#         'zlim' : None,
#         'log' : False,
#         'zoom' : 1,
#         **plot_kwargs
#         }
    
#     ytplot = ytPlot_func(ds, plot_kwargs['normal'], fields)
    
#     if type(plot_kwargs['zlim'])==tuple:
#         ytplot.set_zlim(fields, *plot_kwargs['zlim'])
#     ytplot.set_log(fields, plot_kwargs['log'])
#     ytplot.set_cmap(fields, plot_kwargs['cmap'])
#     ytplot.zoom(plot_kwargs['zoom'])
#     ytplot.annotate_timestamp()
    
#     return ytplot

# # Initialise AxesGrid
# def get_AxesGrid(**grid_kwargs):
#     """
#     Generate figure and grid to plot an array of data.

#     Parameters
#     ----------
#     **grid_kwargs
#         Parameters to change the appearence and format of the grid.

#     Returns
#     -------
#     fig : matplotlib.figure.Figure
#         matplotlib Figure which stores the grid.
#     grid : mpl_toolkits.axes_grid1.axes_grid.ImageGrid
#         The grid object to place data into.
#     """
    
#     assert 'nrows_ncols' in grid_kwargs.keys()
    
#     # Assign default values to create grid.
#     grid_kwargs = {
#         'rect' : (.075,.075,.85,.85),
#         'axes_pad' : .1,
#         'label_mode' : 'L',
#         'share_all' : False,
#         'cbar_location' : 'right',
#         'cbar_mode' : 'single',
#         'cbar_size' : '3%',
#         'cbar_pad' : .1,
#         **grid_kwargs
#         }
    
#     fig = plt.figure()
#     grid = AxesGrid(fig,**grid_kwargs)
    
#     return fig, grid
    
# def plot_YTPlot_grid(ds_ts, fields, ytPlot_func, save:bool, save_dir:str, grid_kwargs:dict={}, plot_kwargs:dict={}):
#     """
#     Plot a grid for a range of datasets. Optimised for yt.SlicePlot.

#     Parameters
#     ----------
#     ds_ts : list
#         List of yt data objects.
#     fields : tuple
#         fields is of the form ("<data structure>","<field name>").
#     ytPlot_func : function
#         ytPlot_func==yt.SlicePlot or ytPlot_func==yt.ProjectionPlot.
#     save : bool
#         Save image of rendering if true, else not.
#     save_dir : str
#         Directory which the rendering is to be saved to.
#     grid_kwargs : dict
#         Parameters to change the appearence and format of the grid.
#     plot_kwargs : dict
#         Parameters to change the appearence of the plot or manipulation of data.

#     Returns
#     -------
#     ytplots : list
#         List of plot objects which can be manipulated.
#     """
    
#     # Get zlim
#     extrema = np.array([d.all_data().quantities.extrema(fields[1]) for d in ds_ts])
#     zlim = np.array(extrema)[:,0].mean(), np.array(extrema)[:,1].mean()
#     plot_kwargs.update({'zlim' : zlim})
    
#     fig, grid = get_AxesGrid(**grid_kwargs)
    
#     ytplots = []
#     for i, d in enumerate(ds_ts):
#         ytplot = plot_ytPlot(d, fields, ytPlot_func, **plot_kwargs)
        
#         # Force the plot to redraw itself on the AxesGrid axes.
#         plot = ytplot.plots[fields]
#         plot.figure = fig
#         plot.axes = grid[i].axes
#         plot.cax = grid.cbar_axes[i]
        
#         # Finally redraw the plot.
#         ytplot._setup_plots()
        
#         # Redraw ticks.
#         ax = grid.axes_all[i]
#         ax.set_xticks(ax.get_xticks()[1:-1])
#         ax.set_yticks(ax.get_yticks()[1:-1])
        
#         ytplots.append(ytplot)
    
#     # Get plot name.
#     import re
#     plot_string = re.findall('[A-Z][^A-Z]*', str(ytPlot_func))[0]
    
#     fig.suptitle(" ".join([f"{plot_string}",f"{ytplot.data_source}".split(': , ')[-1]]), fontsize=20, y=.9)
    
#     if save:
#         plt.savefig(os.path.join(save_dir,f"{plot_string}_{fields[1]}_timeseries.png"),bbox_inches='tight',facecolor='white',dpi=150)
    
#     return ytplots







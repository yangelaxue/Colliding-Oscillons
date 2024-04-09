"""
This file visualises LE data by
- plotting 3D scatter plots of Edens
- animating 3D scatter plots of Edens
- plotting 2D slices of Edens
- animating 2D slices of Edens
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.animation import FuncAnimation, FFMpegWriter
# from IPython import display
# from mpl_toolkits.mplot3d import Axes3D

from utils.latticeeasy import *
from utils.visualisation_utils import scatterfield_3D, get_opacity

#%% Load data

# output_dir = "/media/yangelaxue/TRANSFER/Colliding_Oscillons/CE2"
# output_dir = "/media/yangelaxue/TRANSFER/Colliding_Oscillons/alpha_beta-0.05_100_200.450623/smash_osc41_osc29_0.01c_v1"
# output_dir = "/media/yangelaxue/TRANSFER/Colliding_Oscillons/alpha_beta-0.25_80_351.512634/smash_osc22_osc4_0.01c_v1/larger_box"
output_dir = "/home/yangelaxue/Documents/Uni/Masters/Colliding Oscillons/Test Leon/CE"

data = LATTICEEASY(output_dir)
timelist = [i for i in range(0,len(data.t),1)][::]
# timelist = [0,len(data.t)-1]
print(f"Making plots for times:\n{data.t[timelist]}",)
slices = [slice(0,data.domain_dimensions[0],1),slice(0,data.domain_dimensions[1],1),slice(0,data.domain_dimensions[2],1)]
slices[1] = data.domain_dimensions[1]//2
slices = tuple(slices)

# Edens
plot_Edens_volume = 1
plot_Edens_slice = 1
animate_Edens_volume = False
animate_Edens_slice = False
plot_Edens_bool = (plot_Edens_volume, animate_Edens_volume, plot_Edens_slice, animate_Edens_slice)

n_means_def = 8
plot_EdensRel = False

# phi
plot_phi_slice = 1
animate_phi_slice = False
plot_phi_bool = (plot_phi_slice, animate_phi_slice)
# phip
plot_phip_slice = 0
animate_phip_slice = False
plot_phip_bool = (plot_phip_slice, animate_phip_slice)

#%% Start plotting!

print(f"Making plots for {output_dir}")

#%% Plot Edens

if any((plot_Edens_volume,plot_Edens_slice)):

    print("Making plots of Edens...")

    xyz = [np.linspace(0,width,dim) for width,dim in zip(data.domain_width_pr,data.domain_dimensions)]
    XYZ = np.meshgrid(*xyz,indexing='ij')

    Edens_gen = (data.get_Edens_pr(time) for time in timelist)

    for i, Edens in enumerate(Edens_gen):

        print(f"{timelist[i]} of {data.t.size}.")

        if plot_Edens_volume:

            n_means = n_means_def
            Edens_mean = Edens.mean()

            where = Edens>Edens_mean*n_means
            while where.sum()==0:
                n_means -= 1
                where = Edens>Edens_mean*n_means

            if plot_EdensRel:
                fig, ax, scatter = scatterfield_3D(
                    Edens[where]/Edens_mean,
                    tuple(X[where] for X in XYZ),
                    1/4
                )
                fig.colorbar(scatter, pad=.1, shrink=.6, label=r"$\rho/\langle\rho\rangle$")
            else:
                fig, ax, scatter = scatterfield_3D(
                    Edens[where],
                    tuple(X[where] for X in XYZ),
                    1/4
                )
                fig.colorbar(scatter, pad=.1, shrink=.6, label=r"$\rho$")

            # ax.view_init(90, 0)

            ax.set_xlim(0, data.domain_width_pr[0])
            ax.set_ylim(0, data.domain_width_pr[1])
            ax.set_zlim(0, data.domain_width_pr[2])

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            ax.set_title("Oscillons,\n" + fr"timslice={data.t[timelist[i]]}, $a=${data.a[timelist[i]]}, " + fr"$\rho/\langle\rho\rangle>${n_means}")

            fig.savefig(os.path.join(output_dir,f"Edens volume {data.t[timelist[i]]}.png"))
            plt.close()
        
        if plot_Edens_slice:

            fig_slc = plt.figure(figsize=(6,5),tight_layout=True)
            ax_slc = fig_slc.add_subplot(111)

            mesh = ax_slc.pcolormesh(xyz[0], xyz[1], Edens[slices].T,
                                # vmin=0, vmax=18,
                                )
            cbar = fig_slc.colorbar(mesh, label="Edens (pr)")
            ax_slc.set_xlim(0, data.domain_width_pr[0])
            ax_slc.set_ylim(0, data.domain_width_pr[1])

            ax_slc.set_aspect('equal')
            ax_slc.set_xlabel("x (pr)")
            ax_slc.set_ylabel("z (pr)")
            ax_slc.set_title(f"Edens, time (pr) = {data.t[timelist[i]]}")

            fig_slc.savefig(os.path.join(output_dir,f"Edens slice {data.t[timelist[i]]}.png"))
            plt.close()

if animate_Edens_volume:

    print("Animating Edens...")

    def update_scatter(i):
        
        print(i)
        
        Edens = next(Edens_gen)
        Edens_mean = Edens.mean()
        n_means = n_means_def
        where = Edens>Edens_mean*n_means
        while where.sum()==0:
            n_means -= 1
            where = Edens>Edens_mean*n_means
        
        opacity = get_opacity(Edens)
        
        ax.clear()
        scatter = ax.scatter(X[where], Y[where], Z[where], alpha = 1/4*opacity[where], c=Edens[where],
    #                          vmin=vmin, vmax=vmax
                            )
        
        ax.set_xlim(xyz[0][0],xyz[0][-1])
        ax.set_ylim(xyz[1][0],xyz[1][-1])
        ax.set_zlim(xyz[2][0],xyz[2][-1])
        
        ax.set_xlabel(r'$(x/al_{pl})\times B$')
        ax.set_ylabel(r'$(y/al_{pl})\times B$')
        ax.set_zlabel(r'$(z/al_{pl})\times B$')
        
        ax.set_title(r"$t_{\textrm{pr}}=$"+f"{data.t[i]}" + fr"$\rho/\langle\rho\rangle>${n_means}")
    
    xyz = [np.linspace(0,width,dim) for width,dim in zip(data.domain_width_pr,data.domain_dimensions)]
    XYZ = np.meshgrid(*xyz,indexing='ij')

    Edens_gen = (data.get_Edens_pr(time) for time in timelist)

    Edens_0 = next(Edens_gen)
    Edens_0_mean = Edens_0.mean()

    fig = plt.figure(figsize=(6,6),tight_layout=True)

    ax = fig.add_subplot(111,projection='3d')

    n_means = n_means_def
    where = Edens_0>Edens_0_mean*n_means
    while where.sum()==0:
        n_means -= 1
        where = Edens>Edens_mean*n_means
    opacity = get_opacity(Edens_0)
    scatter = ax.scatter(X[where], Y[where], Z[where], alpha = 1/4*opacity[where], c=Edens_0[where],
                        # vmin=vmin, vmax=vmax
                        )
    # cbar = fig.colorbar(scatter, pad=0.15, shrink=.6)

    ax.set_title(r"$t_{\textrm{pr}}=$"+f"{data.t[timelist[0]]}" + fr"$\rho/\langle\rho\rangle>${n_means}")
    ax.set_xlim(xyz[0][0],xyz[0][-1])
    ax.set_ylim(xyz[1][0],xyz[1][-1])
    ax.set_zlim(xyz[2][0],xyz[2][-1])

    ax.set_xlabel(r'$(x/al_{pl})\times B$')
    ax.set_ylabel(r'$(y/al_{pl})\times B$')
    ax.set_zlabel(r'$(z/al_{pl})\times B$')

    ani = FuncAnimation(fig=fig, func=update_scatter, frames=timelist[1:-1], interval=1000,)
    ani.save(os.path.join(output_dir, "Edens_animation.mp4"), writer=FFMpegWriter(fps=2))

if animate_Edens_slice:

    print("Animating and saving Edens slices.")

    def animate_pcolormesh(i):
    
        print(i)
        Edens = next(Edens_gen)
        mesh.set_array(Edens[slices].T)
        ax.set_title(f"Energy density, time (pr) = {data.t[i]}")

        cbar.remove()
        cbar = fig.colorbar(mesh, label="Edens (pr)")

    xyz = [np.linspace(0,width,dim) for width, dim in zip(data.domain_width_pr, data.domain_dimensions)]

    Edens_gen = (data.generate_Edens_pr(timelist))
    Edens = next(Edens_gen)

    fig = plt.figure(figsize=(6,5),tight_layout=True)
    ax = fig.add_subplot(111)

    mesh = ax.pcolormesh(xyz[0], xyz[2], Edens[slices].T)
    cbar = fig.colorbar(mesh, label="Edens (pr)")

    ax.set_aspect('equal')
    ax.set_xlabel("x (pr)")
    ax.set_ylabel("z (pr)")
    ax.set_title(f"Energy density, time (pr) = {data.t[timelist[0]]}")

    ani = FuncAnimation(fig=fig, func=animate_pcolormesh, frames=timelist[1:-1], interval=1000,)
    ani.save(os.path.join(output_dir, "Edens_slices_animation.mp4"), writer=FFMpegWriter(fps=2))

#%% Plot phi

if any(plot_phi_bool):

    print("Making plots of phi...")

    xyz = [np.linspace(0,width,dim) for width,dim in zip(data.domain_width_pr,data.domain_dimensions)]
    phi_gen = (load_phi(output_dir, data.domain_dimensions, time) for time in timelist)

    for i, phi in enumerate(phi_gen):

        print(f"{timelist[i]} of {data.t.size}.")

        if plot_phi_slice:

            fig = plt.figure(figsize=(6,5),tight_layout=True)
            ax = fig.add_subplot(111)

            mesh = ax.pcolormesh(xyz[0], xyz[1], phi[slices].T,
                                # vmin=-9, vmax=9,
                                )
            cbar = fig.colorbar(mesh, label="phi (pr)")
            ax.set_xlim(0, data.domain_width_pr[0])
            ax.set_ylim(0, data.domain_width_pr[1])

            ax.set_aspect('equal')
            ax.set_xlabel("x (pr)")
            ax.set_ylabel("z (pr)")
            ax.set_title(f"phi, time (pr) = {data.t[timelist[i]]}")

            plt.savefig(os.path.join(output_dir,f"phi slice {data.t[timelist[i]]}.png"))
            plt.close()

#%% Plot Edens

if any(plot_phip_bool):

    print("Making plots of phip...")

    xyz = [np.linspace(0,width,dim) for width,dim in zip(data.domain_width_pr,data.domain_dimensions)]
    phip_gen = (load_phip(output_dir, data.domain_dimensions, time) for time in timelist)

    for i, phip in enumerate(phip_gen):

        print(f"{timelist[i]} of {data.t.size}.")

        if plot_phip_slice:

            fig = plt.figure(figsize=(6,5),tight_layout=True)
            ax = fig.add_subplot(111)

            mesh = ax.pcolormesh(xyz[0], xyz[1], phip[slices].T,
                                # vmin=-5, vmax=5,
                                )
            cbar = fig.colorbar(mesh, label="phi (pr)")
            ax.set_xlim(0, data.domain_width_pr[0])
            ax.set_ylim(0, data.domain_width_pr[1])

            ax.set_aspect('equal')
            ax.set_xlabel("x (pr)")
            ax.set_ylabel("z (pr)")
            ax.set_title(f"phip, time (pr) = {data.t[timelist[i]]}")

            plt.savefig(os.path.join(output_dir,f"phip slice {data.t[timelist[i]]}.png"))
            plt.close()













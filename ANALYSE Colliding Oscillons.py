"""
Script to analyse oscillon collisions. Particularly, their velocities.
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import os, pickle
from scipy.integrate import simpson

from utils.latticeeasy import *
from utils.label_utils import label_blobs, get_labels
from utils.oscillons import *

#%% Adjustable parameters

# output_dir = "/media/yangelaxue/TRANSFER/Colliding_Oscillons/alpha_beta-0.25_80_351.512634/smash_osc22_osc4_0.01c_v1"
output_dir = "/media/yangelaxue/TRANSFER/Colliding_Oscillons/alpha_beta-0.75_100_301.15863/smash_osc12_osc2_0.001c_v1"
# output_dir = "/media/yangelaxue/TRANSFER/Colliding_Oscillons/CE1"
output_main_dir = "/media/yangelaxue/TRANSFER/Colliding_Oscillons/alpha_beta-0.75_100"
t_pr = 301.15863
# speeds = "0.01c"

WHOLE_FIELD = False
n_means = 2
CALC_COMS = True
CALC_VELOCITIES = True
CALC_EXPECTED_VELOCITY = False
CALC_MASSES = True
if WHOLE_FIELD:
    f_ext = "_field"
else:
    f_ext = ""

data = LATTICEEASY(output_dir)

# output_main_dir = '_'.join('/'.join(output_dir.split('/')[:-1]).split('_')[:-1])
data_main = LATTICEEASY(output_main_dir)

timeslices = [i for i in range(data.t.size)][:3:]

#%% Load mean energy density threshold

# t_pr = float(output_dir.split('/')[-2].split('_')[-1])
timeslice, = np.where(data_main.t==t_pr)[0]
Edens_mean = load_energy(output_main_dir,timeslice)[1:].sum()

#%% Define some parameters

assert all(data_main.a[timeslice]==data.a)
a = data.a[0]
A, B, r, s = data.rescale_A, data.rescale_B, data.rescale_r, data.rescale_s
c_pr = 1/(a**(data.rescale_s+1))
key = "alpha_beta-{}_{}".format(round(data.alpha,2),int(data.beta))

#%% Label oscillons.

oscillons_fs = os.path.join(output_dir,"oscillons_dict_{}.p")

if not WHOLE_FIELD:
    for timeslice in timeslices:
        oscillons_f = oscillons_fs.format(data.t[timeslice])
        if not os.path.exists(oscillons_f):
            print(f"Finding oscillons for timeslice {timeslice}/{timeslices[-1]}")
            Edens = data.get_Edens_pr(timeslice)
            oscillons = label_blobs(Edens, Edens_mean*n_means, 134, True)
            pickle.dump(oscillons, open(oscillons_f, 'wb'))

#%% Calculate Center of Mass positions

if CALC_COMS:
    print("Calculating Center of Mass Positions.")
    if os.path.exists(os.path.join(output_dir,f"coms_t_dict{f_ext}.p")):
        coms_dict = pickle.load(open(os.path.join(output_dir,f"coms_t_dict{f_ext}.p"), 'rb'))
    else:
        coms_dict = {}

    calc_timeslices = []
    for timeslice in timeslices:
        if not data.t[timeslice] in coms_dict:
            calc_timeslices.append(timeslice)

    if len(calc_timeslices)==0:
        print("Center of mass positions already saved.")
    else:
        Edens_t = (data.get_Edens_pr(timeslice) for timeslice in calc_timeslices)
        if WHOLE_FIELD:
            xyz = np.array([x*width/dim for x, width, dim in zip(np.indices(data.domain_dimensions),data.domain_width_pr,data.domain_dimensions)])
            coms_t = [[np.array([(Edens*x).sum() for x in xyz])/Edens.sum()] for Edens in Edens_t]
        else:        
            oscillons_t = (pickle.load(open(oscillons_fs.format(data.t[timeslice]),'rb')) for timeslice in calc_timeslices)
            coms_t = [get_oscillon_coms(Edens,data.domain_width_pr,oscillons,None)[1] for Edens,oscillons in zip(Edens_t,oscillons_t)]
        coms_dict.update({data.t[timeslice]:coms for timeslice,coms in zip(calc_timeslices,coms_t)})
        pickle.dump(coms_dict, open(os.path.join(output_dir,f"coms_t_dict{f_ext}.p"),'wb'))

if CALC_VELOCITIES:
    print("Calculating velocities (for single oscillons only)")

    coms_dict = pickle.load(open(os.path.join(output_dir,f"coms_t_dict{f_ext}.p"), 'rb'))
    coms_t = [coms_dict[data.t[timeslice]][0] for timeslice in timeslices]
    # print(coms_t)
    com_velocity_t = np.diff(coms_t,axis=0)/np.tile(np.diff(data.t[timeslices]),(3,1)).T
    com_velocity_dict = {data.t[timeslice]:com_velocity for timeslice,com_velocity in zip(timeslices,com_velocity_t)}
    pickle.dump(com_velocity_dict, open(os.path.join(output_dir,f"com_velocities_t_dict{f_ext}.p"),'wb'))

if CALC_EXPECTED_VELOCITY:

    if os.path.exists(os.path.join(output_dir,f"expected_velocities_t_dict{f_ext}.p")):
        expected_velocity_dict = pickle.load(open(os.path.join(output_dir,f"expected_velocities_t_dict{f_ext}.p"),'rb'))
        print(expected_velocity_dict)
    else:
        expected_velocity_dict = {}

    calc_timeslices = []
    for timeslice in timeslices:
        if not data.t[timeslice] in expected_velocity_dict:
            calc_timeslices.append(timeslice)

    if len(calc_timeslices)==0:
        print("Expected velocities already saved.")
    else:
        expected_velocity_t = np.array([data.get_field_speed_pr(timeslice) for timeslice in calc_timeslices])
        expected_velocity_dict.update({data.t[timeslice]:expected_velocity for timeslice,expected_velocity in zip(calc_timeslices,expected_velocity_t)})
        pickle.dump(expected_velocity_dict, open(os.path.join(output_dir,f"expected_velocities_t_dict{f_ext}.p"),'wb'))

if CALC_MASSES:
    print("Calculating Maases of Oscillons.")

    if os.path.exists(os.path.join(output_dir,f"masses_t_dict{f_ext}.p")):
        print("True")
        masses_dict = pickle.load(open(os.path.join(output_dir,f"masses_t_dict{f_ext}.p"), 'rb'))
    else:
        masses_dict = {}

    calc_timeslices = []
    for timeslice in timeslices:
        if not data.t[timeslice] in masses_dict:
            calc_timeslices.append(timeslice)

    if len(calc_timeslices)==0:
        print("Masses already saved.")
    else:
        Edens_t = (data.get_Edens_pr(timeslice) for timeslice in calc_timeslices)
        if WHOLE_FIELD:
            oscillons_t = (np.zeros(data.domain_dimensions) for timeslice in calc_timeslices)
        else:        
            oscillons_t = (pickle.load(open(oscillons_fs.format(data.t[timeslice]),'rb')) for timeslice in calc_timeslices)
        masses_t = [get_oscillon_masses(Edens, data.domain_width_pr, oscillons, labels=None) for Edens,oscillons in zip(Edens_t,oscillons_t)]
        masses_dict.update({data.t[timeslice]:masses for timeslice,masses in zip(calc_timeslices,masses_t)})
        pickle.dump(masses_dict, open(os.path.join(output_dir,f"masses_t_dict{f_ext}.p"),'wb'))


# %%

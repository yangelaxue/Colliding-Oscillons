"""
Script to analyse oscillon collisions. Particularly, their velocities.
"""

#%% Imports

import numpy as np
import os, pickle

from utils.latticeeasy import *
from utils.label_utils import label_blobs
from utils.oscillons import *

#%% Adjustable parameters

output_main_dir = "/media/yangelaxue/23E7CCB1624D2A50/Colliding_Oscillons/alpha_beta-0.05_25"
output_dir = "/media/yangelaxue/23E7CCB1624D2A50/Colliding_Oscillons/alpha_beta-0.05_25_695.675659/trial_2/smash_osc7_128_osc7_128_0.02"

# From output_dirs, find t_pr.
idx = len([1 for s,t in zip(output_main_dir, output_dir) if s==t])
t_pr = float(output_dir[idx+1:].split('/')[0])

# Which quanitities to calculate and how.
WHOLE_FIELD = 0
n_means = .5
CALC_COMS = True
CALC_VELOCITIES = True
CALC_EXPECTED_VELOCITY = False
CALC_MOMENTUM_SUM = False
CALC_MASSES = True
if WHOLE_FIELD:
    f_ext = "_field"
else:
    f_ext = ""

# Load data
data = LATTICEEASY(output_dir)
data_main = LATTICEEASY(output_main_dir)
timeslices = [i for i in range(data.t.size)][450::2]

#%% Load mean energy density threshold

timeslice, = np.where(data_main.t==t_pr)[0]
Edens_mean = load_energy(output_main_dir,timeslice)[1:].sum()

#%% Define some parameters

print(data_main.a[timeslice])
print(data.a[timeslices])
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

    print(data.t[calc_timeslices])

    if len(calc_timeslices)==0:
        print("Center of mass positions already saved.")
    else:
        Edens_t = (data.get_Edens_pr(timeslice) for timeslice in calc_timeslices)
        if WHOLE_FIELD:
            xyz = np.array([x*width/dim for x, width, dim in zip(np.indices(data.domain_dimensions),data.domain_width_pr,data.domain_dimensions)])
            coms_t = [[np.array([(Edens*x).sum() for x in xyz])/Edens.sum()] for Edens in Edens_t]
        else:        
            print(data.t[timeslice])
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
    print("Calculating expected velocities (of the whole field)")

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

if CALC_MOMENTUM_SUM:
    print("Calculating total field momentum")

    if os.path.exists(os.path.join(output_dir,f"momentum_sum_t_dict{f_ext}.p")):
        momentum_sum_t_dict = pickle.load(open(os.path.join(output_dir,f"momentum_sum_t_dict{f_ext}.p"),'rb'))
        print(momentum_sum_t_dict)
    else:
        momentum_sum_t_dict = {}

    calc_timeslices = []
    for timeslice in timeslices:
        if not data.t[timeslice] in momentum_sum_t_dict:
            calc_timeslices.append(timeslice)

    if len(calc_timeslices)==0:
        print("Total field momentum already saved.")
    else:
        momentum_sum_t = np.array([data.get_momenta_pr(timeslice).sum(axis=(1,2,3)) for timeslice in calc_timeslices])
        momentum_sum_t_dict.update({data.t[timeslice]:momentum_sum for timeslice,momentum_sum in zip(calc_timeslices,momentum_sum_t)})
        pickle.dump(momentum_sum_t_dict, open(os.path.join(output_dir,f"momentum_sum_t_dict{f_ext}.p"),'wb'))

if CALC_MASSES:
    print("Calculating Masses of Oscillons.")

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
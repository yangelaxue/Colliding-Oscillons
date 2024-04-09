import numpy as np
import os, pickle

from utils.latticeeasy import *

#%% Start findind com speed and momentum speed.

def save_slices(output_dir):

    # Edens_slices = []
    if os.path.exists(os.path.join(output_dir,"Edens slices.p")):
        Edens_slices = pickle.load(open(os.path.join(output_dir,"Edens slices.p"), "rb"))[1]
    else:
        Edens_slices = []

    data = LATTICEEASY(output_dir)
    timeslices = [i for i in range(len(Edens_slices),data.t.size,1)]
    res = data.domain_dimensions[0]
    slices = tuple([slice(0,res),res//2,slice(0,res)])

    print(len(Edens_slices), timeslices)

    print(f"Saving Edens slices for {output_dir = }")

    xyz = np.array([x*width/dim for x, width, dim in zip(np.indices(data.domain_dimensions),data.domain_width_pr,data.domain_dimensions)])
    Edens_gen = (data.get_Edens_pr(timeslice) for timeslice in timeslices)

    for i, (timeslice, Edens) in enumerate(zip(timeslices, Edens_gen)):
        
        print(f"{timeslice} of {timeslices[-1]}")

        Edens_slices.append(Edens[slices])

        pickle.dump([data.t[:len(Edens_slices)+i+1], Edens_slices], open(os.path.join(output_dir,"Edens slices.p"), "wb"))

output_dirs = [
    "/media/yangelaxue/23E7CCB1624D2A50/Colliding_Oscillons/alpha_beta-0.5_50_250.804626/test1/smash_peak+0_peak+0_0.01c_v1",
    "/media/yangelaxue/23E7CCB1624D2A50/Colliding_Oscillons/alpha_beta-0.5_50_250.804626/test1/smash_peak+0_peak+1_0.01c_v1",
    "/media/yangelaxue/23E7CCB1624D2A50/Colliding_Oscillons/alpha_beta-0.5_50_250.804626/test1/smash_peak+0_peak+2_0.01c_v1",
    "/media/yangelaxue/23E7CCB1624D2A50/Colliding_Oscillons/alpha_beta-0.5_50_250.804626/test1/smash_peak+0_peak+3_0.01c_v1",
    "/media/yangelaxue/23E7CCB1624D2A50/Colliding_Oscillons/alpha_beta-0.5_50_250.804626/test1/smash_peak+0_peak+4_0.01c_v1",
    "/media/yangelaxue/23E7CCB1624D2A50/Colliding_Oscillons/alpha_beta-0.5_50_250.804626/test1/smash_peak+0_peak+5_0.01c_v1",
    "/media/yangelaxue/23E7CCB1624D2A50/Colliding_Oscillons/alpha_beta-0.5_50_250.804626/test1/smash_peak+0_peak+6_0.01c_v1",
    "/media/yangelaxue/23E7CCB1624D2A50/Colliding_Oscillons/alpha_beta-0.5_50_250.804626/test1/smash_peak+0_peak+7_0.01c_v1",
]

def main():

    for output_dir in output_dirs:
        save_slices(output_dir)

if __name__=="__main__":
    main()
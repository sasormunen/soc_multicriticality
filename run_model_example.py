# +

import numpy as np
from matplotlib import pyplot as plt
import math
from helper_functions import *
from scipy.integrate import odeint, RK45, solve_ivp
import networkx as nx
from random import choice as rchoice
from random import random as rrandom
from numpy.random import randint
import time
import csv
import os
from model_euler import *
import pandas as pd
import sys



##MODEL PARAMETERS
param_a = 0.8
param_b = 10.5
param_c = 1
param_d = None


#COUPLING MATRIX

d1 = -1.4 
d2 = 0.3 
d3 = -6.8
d4 = 0.9 

diffmat = np.asarray([[d1,d2],[d3,d4]])


#INITIAL CONDITIONS
initial = "uniform" #"half-random"
initial_mat = None #initial conditions can be given directly with this
U_s = 0.01
V_s = 0.01


#DRIFT PARAMETERS
diffusion_on = True
tstep = 5000 #parameter s defining time between topology updates
end = 8000000000 #length of simulation
n_tsteps = int(end/tstep)

track_parameters = True 
edgetrack = True #for writing out all edgechanges in the network
writeout = True
recording = None #can be given a list of three nodes, whose values of U and V will be tracked

turing_update = True
reverse_turing = True

mod_val = 0.001
increment_turing = 0.1*mod_val #0.00005 #0.00002 #0.000004 #0.000004#0.01 #0.02 for small 3-node, 0.01 for others
remov_turing = mod_val #0.00002 #0.00001 #0.00002 #0.01 #0.02
turing_threshold = 0.005

hopf_update = True
increment_hopf = mod_val #0.00006 #0.001 #0.005
remov_hopf = mod_val #0.0004 #0.00004 #0.000024 #0.0004 #0.01
hopf_threshold = 0.05 #0.1


noise = 0.01 #after each integration window, we add random uniform noise from interval -0.01 to 0.01

count = 10

pad = 0.0001
min_link_weight = 0.0001


#final runs
name = "ER_N_100_edgevar_avgdeg_8_startlam_16.37" 


cont2 = False

if cont2 == True:
    name = "fn_uniformrises_rnoise_ER_N_100_edgevar_avgdeg_8_startlam_16.37_cmat_-1.4_0.3_-6.8_0.9_step_5000_turing_ON_revTrue_hopf_ON_tinc_0.0001_tremov_0.001_hinc_0.001_hremov_0.001_count_10_htheta_0.05_ttheta_0.005"
origname = name

if cont2 == True:

    #check which conts there are already
    tpath = "../files/timeseries/"

    for contind in np.arange(2,100):
        isFile = os.path.isfile(tpath + name + "_cont" + str(contind)+".csv")
        if isFile == False:
            break

    
    name = name + "_cont" + str(contind)


    startind = name.index("cmat")+5
    endind = name.index("_step_")
    string = name[startind:endind]
    string = string.split("_")
    d1 = float(string[0])
    d2 = float(string[1])
    d3 = float(string[2])
    d4 = float(string[3])
    diffmat = np.asarray([[d1,d2],[d3,d4]])

    ind1 = name.index("step_")
    ind2 = name.index("_turing_")
    tstep = int(name[ind1 + 5 : ind2])
    
    ind1 = name.index("tinc")
    ind2 = name.index("_tremov")
    increment_turing = float(name[ind1 + 5 : ind2])
    ind1 = name.index("tremov")
    ind2 = name.index("_hinc")
    remov_turing = float(name[ind1 + 7 : ind2])
    
    
    ind1 = name.index("ttheta")
    ind2 = name.index("_cont")
    turing_threshold = float(name[ind1 + 7 : ind2])
    
    ind1 = name.index("hinc")
    ind2 = name.index("_hremov")
    increment_hopf = float(name[ind1 + 5 :ind2])
    ind1 = name.index("hremov")
    ind2 = name.index("_count")
    remov_hopf = float(name[ind1 + 7 :ind2])
    ind1 = name.index("htheta")
    ind2 = name.index("_ttheta")
    hopf_threshold = float(name[ind1 + 7 : ind2])
    ind1 = name.index("count")
    ind2 = name.index("_htheta")
    count = int(name[ind1 + 6 : ind2])
    

###BELOW DOES NOT NEED TO BE MODIFIED

if cont2 == False:
    path = "../files/static_graphs/"
    G = read_graph_correct(path,name)
    N = len(G.nodes)
else:
    #get the previous files
    prevfiles = [origname]
    for i in np.arange(2,contind):
        prevfiles.append(origname + "_cont" + str(i))

    df = pd.read_csv("../files/timeseries/"+prevfiles[-1] + ".csv", sep='\s+', header=None)
    last_tval = list(df[0])[-1]
    G = read_edgechanges(prevfiles[-1], [last_tval], writeout=True, outpath = "../files/evolved_edgelists/")
    N = len(G.nodes)
    #read initial mat which otherwise is None
    
    df = pd.read_csv("../files/states/"+prevfiles[-1] + ".csv", sep='\s+', header=None)
    vec  = df[0]
    initial_mat = np.reshape(vec, (N, 2))
    


hopfval, turingval = "OFF", "OFF"
if hopf_update == True:
    hopfval = "ON"
if turing_update == True:
    turingval = "ON"

if reverse_turing == True:
    reversepart = "_revTrue_"
else:
    reversepart = "_revFalse_"

# noisemat would be added to the uniform initial conditions, but this is not in use currently
#NOISE FOR INITIAL CONDITIONS (not used after that)
#amplitude_initial_noise = 0.01 
#N = len(G.nodes)
#noise1 = np.random.uniform(low=-amplitude_initial_noise, high=amplitude_initial_noise, size=N)
#noise2 = np.random.uniform(low=-amplitude_initial_noise, high=amplitude_initial_noise, size=N)
#noisemat = np.column_stack((noise1,noise2))


#this is how all the produced files will be named
if cont2 == False:
    graphname = name + "_cmat_" + str(d1) + "_"+ str(d2) + "_" + str(d3) + "_" + str(d4) 
    graphname  = "fn_uniformrises_rnoise_" + graphname + "_step_" + str(tstep) + "_turing_"+turingval + reversepart + "hopf_" + hopfval + "_tinc_" +str(increment_turing) + "_tremov_" + str(remov_turing) + "_hinc_"+ str(increment_hopf) + "_hremov_"+ str(remov_hopf) + "_count_" + str(count) + "_htheta_" + str(hopf_threshold) + "_ttheta_" + str(turing_threshold)
else:
    indd = name.index("_cmat")
    checkname = name[:indd] + "_cmat_" + str(d1) + "_"+ str(d2) + "_" + str(d3) + "_" + str(d4)  + "_step_" + str(tstep) + "_turing_"+turingval + reversepart + "hopf_" + hopfval + "_tinc_" +str(increment_turing) + "_tremov_" + str(remov_turing) + "_hinc_"+ str(increment_hopf) + "_hremov_"+ str(remov_hopf) + "_count_" + str(count) + "_htheta_" + str(hopf_threshold) + "_ttheta_" + str(turing_threshold)
    if checkname != origname:
        print("mismatch in names:", origname, checkname)
        sys.exit()
    graphname = name



#G = Gmod
rm = RAModel(G, n_tsteps = n_tsteps, initial=initial, tstep=tstep, turing_threshold = turing_threshold, turing_update = turing_update, 
             increment_turing = increment_turing, track_parameters = track_parameters, hopf_update=hopf_update,
             increment_hopf=increment_hopf, remov_hopf=remov_hopf, diffusion_on = diffusion_on, hopf_threshold = hopf_threshold ,
             graphname = graphname, writeout=writeout, noise=noise,
             remov_turing = remov_turing, U_s = U_s, V_s = V_s, initial_mat = initial_mat,
              edgetrack=edgetrack,count=count,investigate_divergence=False,final_mat=None,
             param_a=param_a,param_b=param_b,param_c=param_c, param_d=param_d, diffmat=diffmat,reverse_turing = reverse_turing, pad = pad, min_link_weight =min_link_weight, recording=recording)


rm.simulate()


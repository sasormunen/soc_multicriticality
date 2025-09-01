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
from model import *
import pandas as pd
import sys



##Parameters of the FitzHugh-Nagumo equations
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
initial = "uniform" #options: "uniform", "half-random", "random"
initial_mat = None #initial conditions can be given directly through this matrix
U_s = 0.01 #the initial values
V_s = 0.01


#DRIFT PARAMETERS
diffusion_on = True
tstep = 5000 #parameter s defining time between topology updates
end = 100000 #8000000000 #length of simulation
n_tsteps = int(end/tstep)

track_parameters = True #calculates network parameters at each topology update
write_every = 1 #writes to file after len of list ==write_every
edgetrack = True #for writing out all edgechanges in the network so that the topological evolution can be reproduced
writeout = True
recording = None #can be given a list of three nodes, whose values of U and V will be tracked

turing_update = True
reverse_turing = True #this should be True to obtain the behavior described in the paper

mod_val = 0.001 #how much edge weights are changed in the updates (parameter delta w in the article)
beta = 0.1 #controls time-scale separation between Turing and Hopf subcriticality updataes
increment_turing = beta*mod_val 
remov_turing = mod_val #0.00002 #0.00001 #0.00002 #0.01 #0.02
turing_threshold = 0.005

hopf_update = True
increment_hopf = mod_val #0.00006 #0.001 #0.005
remov_hopf = mod_val #0.0004 #0.00004 #0.000024 #0.0004 #0.01
hopf_threshold = 0.05 #0.1


noise = 0.01 #after each integration window, we add random uniform noise from interval -0.01 to 0.01

count = 10 #controls time-scale separation between sub-and supercriticality updates (parameter c in the article)

pad = 0.0001 #parameter p for when the alternative scaling rule is used (see SI VII)
min_link_weight = 0.0001


#final runs
name = "ER_N_100_edgevar_avgdeg_8_startlam_16.37" 
origname = name

###BELOW DOES NOT NEED TO BE MODIFIED


path = "../files/static_graphs/"
G = read_graph_correct(path,name)
N = len(G.nodes)

hopfval, turingval = "OFF", "OFF"
if hopf_update == True:
    hopfval = "ON"
if turing_update == True:
    turingval = "ON"

if reverse_turing == True:
    reversepart = "_revTrue_"
else:
    reversepart = "_revFalse_"

#this is how all the produced files will be named

graphname = name + "_cmat_" + str(d1) + "_"+ str(d2) + "_" + str(d3) + "_" + str(d4) 
graphname  = "fn_uniformrises_rnoise_" + graphname + "_step_" + str(tstep) + "_turing_"+turingval + reversepart + "hopf_" + hopfval + "_tinc_" +str(increment_turing) + "_tremov_" + str(remov_turing) + "_hinc_"+ str(increment_hopf) + "_hremov_"+ str(remov_hopf) + "_count_" + str(count) + "_htheta_" + str(hopf_threshold) + "_ttheta_" + str(turing_threshold)


#G = Gmod
rm = RAModel(G, n_tsteps = n_tsteps, initial=initial, tstep=tstep, turing_threshold = turing_threshold, turing_update = turing_update, 
             increment_turing = increment_turing, track_parameters = track_parameters, hopf_update=hopf_update,
             increment_hopf=increment_hopf, remov_hopf=remov_hopf, diffusion_on = diffusion_on, hopf_threshold = hopf_threshold ,
             graphname = graphname, writeout=writeout, noise=noise,
             remov_turing = remov_turing, U_s = U_s, V_s = V_s, initial_mat = initial_mat,
              edgetrack=edgetrack,count=count,investigate_divergence=False,final_mat=None,
             param_a=param_a,param_b=param_b,param_c=param_c, param_d=param_d, diffmat=diffmat,reverse_turing = reverse_turing, pad = pad, min_link_weight =min_link_weight, recording=recording, write_every = write_every)


rm.simulate()


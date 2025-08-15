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
import random
import sys
import os
import pandas as pd



def read_edgechanges(fname,tvals,writeout=True,outpath = "../files/evolved_edgelists/"):

    """ evolve the graph according to edgechanges-file and write out the graph at given timesteps
        If writeout ==False, searches for the first t in tvals and returns avgdeg at that point"""
    
    inpath = "../files/edgechanges/"

    for fil in os.listdir(inpath):
        if fil.startswith(fname + "_tstart"): # and (fend in fil):
            graphfile = fil
            break

    G = read_graph_correct(inpath,graphfile[:-9])
    N = len(G.nodes)
    
    df = pd.read_csv(inpath +fname + ".csv", sep='\s+', names=["t","diff","edge"], header=None)
    times,diffs,edges = df["t"],df["diff"],df["edge"]
    #print(sys.getsizeof(edges))
    #print(f&quot;Size of the list: {sys.getsizeof(edges)} bytes&quot;)

    
    record_ind = 0
    next_record = tvals[record_ind]
    
    len_times,len_tvals = len(times), len(tvals)
    
    i = 0
    while i < len_times and record_ind < len_tvals:
        t,diff,edge = float(times[i]),float(diffs[i]),edges[i]
        if t > next_record: #this needs to be done before the next action
            if writeout == True:
                outfile = outpath + fname + "_t_" + str(next_record) + ".edgelist"
                nx.write_weighted_edgelist(G, outfile)
                
            record_ind += 1
            if record_ind == len_tvals:
                break
            next_record = tvals[record_ind]

        if edge == "increment":
            #add the amount to all
            for n1,n2,d in G.edges(data=True):
                G[n1][n2]['weight'] += diff
        else:
            n1,n2=int(edge[edge.index("(")+1:edge.index(",")]),int(edge[edge.index(",")+1:edge.index(")")])
            G[n1][n2]['weight'] += diff #because it should be minus signed when removing
            
        i += 1

    #varotoimi
    if t < next_record:
        print("timeseries ends at next_record!!!!")
        return G #, times,diffs,edges

    #print(G[0])

    return G #, times,diffs,edges
    


def return_SCC(G):

    N = len(G.nodes())
    
    largest_cc = max(nx.connected_components(G), key=len)

    #if len(largest_cc) != N:
    G = G.subgraph(largest_cc).copy()
    existing_nodes = list(G.nodes)
    #random.shuffle(existing_nodes)
    N = len(existing_nodes)
    nodes_new = np.arange(0,N)

    di = {old_label:new_label for new_label, old_label in enumerate(existing_nodes)}
    G = nx.relabel_nodes(G, di, copy=True) 
        
    return G



def create_network(avgdeg,n,graphtype = "small",rewiring = 0):
    
    """ small custom, ER or lattice (note that n is the dim of lattice, not the number of nodes like for ER!) """
    
    if graphtype == "small":
        
        G = nx.Graph()
        G.add_nodes_from([0,1,2])
        G.add_edge(0,1,weight=1.)
        G.add_edge(1,2,weight=1.)
        G.add_edge(2,0,weight=1.)
        #G.add_edges_from([(0,1),(1, 2), (2, 0)])
        graphname = "threenode"
    
    elif graphtype == "ER":
        G,_ = ER_graph_undirected(avg_deg = avgdeg,n = n)
        graphname = "ER"

    elif graphtype == "ring":

        #if n%2 == 0 :
            #print("n should be odd!!!")
            #return None

        G = nx.Graph()
        nodes = np.arange(n)
        G.add_nodes_from(nodes)
        for i in range(n-1):
            G.add_edge(i,i+1,weight=1.)
        G.add_edge(n-1,0,weight = 1.)
        graphname = "ring"

        if rewiring > 0:
            double = np.concatenate((nodes,nodes))
    
            for i in range(n):
                #print(i)
                n2 = double[rewiring+i]
                if not (G.has_edge(n2,i) or G.has_edge(i,n2)):
                    G.add_edge(i,n2, weight = 1.)

            graphname = "ring_rewired_"+str(rewiring)
        
    elif graphtype == "lattice":

        d = math.floor(math.sqrt(n))
        
        grid_G = nx.grid_2d_graph(d,d, periodic=True)
        grid_nodes = grid_G.nodes
        grid_edges = grid_G.edges
        
        G = nx.Graph()
        G.add_nodes_from(grid_nodes)
        for edge in grid_edges:
            G.add_edge(edge[0],edge[1],weight=1.)
        graphname = "lattice"

    #only returning the SCC, takes care of node labeling as well
    G = return_SCC(G)
    
    return G, graphname


def leading_eigval(mat):
    
    eigvals,eigvecs = np.linalg.eig(mat)
    leading_eig = max(eigvals)
    
    return leading_eig
        


def laplacian(G):
    
    """ returns the graph laplacian"""
    
    L = nx.laplacian_matrix(G).toarray()
    L = L.astype(float) #to makes sure that we can add link_increments that are not integers
    
    return L



def eigenvalues_of_laplacian(G):

    """ left or right eigenvectors, doesn't matter if I just need the vals and note the vectors """

    L = nx.laplacian_matrix(G).toarray()
    vals, vecs = np.linalg.eig(L)
    return vals,vecs




def modify_graph(G, action, change):

    if action == "increase":

        for n1,n2,d in G.edges(data=True):
            d['weight']+= change

    elif action == "decrease":
        
        for n1,n2,d in G.edges(data=True):
            
            if d['weight'] < change:
                print("NOOO,negative edgeweights!!!")
                return
                
            d['weight']-= change
    else:
        print("FAIL: wrong argument!!!")
        return

    return G


def write_graph_file(G,name):

    folder = "../files/graph_files/"
    nx.write_weighted_edgelist(G, folder + name +".edgelist")








    
class RAModel():
    
    """ """  
        
    
    def __init__(self, G, n_tsteps, initial, tstep, turing_threshold, turing_update,
                increment_turing, track_parameters, hopf_update, increment_hopf, remov_hopf, diffusion_on, hopf_threshold,
                graphname, writeout, noise, remov_turing, U_s, V_s, 
                initial_mat, edgetrack,count, reverse_turing, investigate_divergence = False, final_mat = [], param_a = None, param_b =
                 None, param_c = None, param_d = None, diffmat=None, pad = 0.01, min_link_weight =
                0.0001,recording = []):

        self.G = G
        self.graphname = graphname
        self.nodes = np.asarray(sorted(list(G.nodes)))
        self.N = G.number_of_nodes()
        self.n_edges = len(G.edges())

        i = 0
        for node in G.nodes():
            if not self.nodes[i] == node:
                print("not sorted!!!!")
                self.G = 0 #this should result in a fail
                n_tsteps = 0
            i += 1
            
        self.L = laplacian(G) 

        self.Cmat = diffmat
        
        self.initial = initial
        #self.noisemat = noisemat
        self.turing_update = turing_update
        self.diffusion_on = diffusion_on
        self.count = count
        
        self.turing_threshold  = turing_threshold
        self.reverse_turing = reverse_turing
        
        self.n_tsteps = n_tsteps
        self.tstep = tstep
        self.recording = recording
        
        self.U_s = U_s
        self.V_s = V_s

        self.increment_turing = increment_turing
        self.remov_turing = remov_turing
        self.min_link_weight =  min_link_weight #3.0 #0.001 #arbitrary choice at the moment
        
        self.track_parameters = track_parameters
        self.writeout = writeout
        self.hopf_update = hopf_update
        self.increment_hopf = increment_hopf
        self.remov_hopf = remov_hopf
        self.hopf_threshold = hopf_threshold


        self.noise_amplitude = noise
        self.initial_mat = initial_mat

        self.a = param_a
        self.b = param_b
        self.c = param_c
        self.d = param_d

        self.pad = pad
        

        #parameters to track
        self.tracking_tvals = []
        self.avg_degs = []
        self.lambdas = []
        self.secondsmallest = []
        self.n_violating_turing = []
        self.n_violating_hopf= []  
        self.fracs_tsup_restrict = []
        self.fracs_hsub_restrict = []
 
        self.adds = 0
        self.removs = 0

        self.effects_remov = []
        self.effects_add = []


        self.modification_ts = []
        self.modifications = []
        self.mod_edges = []
        

        self.edgetrack = edgetrack

        self.investigate_divergence = investigate_divergence
        self.final_mat = final_mat
        self.final_reached = None 


        self.stdU = 0
        self.stdV = 0

            
        self.wdict = {}

        self.weight_stds = []
        self.overlaps = []
        self.turing_stdUs = []
        self.turing_stdVs = []
        self.max_amplitudes_U = []
        self.max_amplitudes_V = []
        self.eigcents_all = []
        self.eigcent_stds = []
        self.eigcents_hopf = []
        self.eigcents_turing = []
        self.eigcents_hopf_unweighted = []
        self.eigcents_turing_unweighted = []
        self.effects_hopf_sub = []
        self.effects_hopf_super = []
        self.effects_turing_sub = []
        self.effects_turing_super = []

        self.effects_hopf_sub_kn = []
        self.effects_hopf_super_kn = []
        self.effects_turing_sub_kn = []
        self.effects_turing_super_kn = []
                


    def dynamics_HO(self, t, x): #, args):

        #this returns the derivatives in the form [dUdt_1 dVdt_1 dUdt_2, dVdt_2]
        #higher order integration 
        #this argument t is here just because I needed it for the solve_ivp function to work

        #x is now a vector -> put back to a matrix
        mat = np.reshape(x, (self.N, 2))
            
        U,V = mat[:,0],mat[:,1]

        dUdt = self.c*(-U**3 + U - V) - self.Cmat[0,0] * self.L @ U - self.Cmat[0,1] * self.L @ V
        dVdt = self.c * self.b* (U -self.a*V) - self.Cmat[1,1]*self.L @ V - self.Cmat[1,0]*self.L @ U
        #dUdt = self.r*U*(1-U/self.k) - self.q*U*V/(self.W + U) - self.Cmat[0,0] * self.L @ U - self.Cmat[0,1] * self.L @ V
        #dUdt = (U**2+6*U+1)/self.k- self.q*U*V/(W + U) - self.Cmat[0,0] * self.L @ U + self.Cmat[0,1] * self.L @ V
        #dVdt = self.eta*self.q*U*V/(self.W + U) - self.M*V**2 - self.Cmat[1,1]*self.L @ V - self.Cmat[1,0]*self.L @ U
        
        #flatten back so that I get a list [dU1dt, dV1dt, dU2dt, dVdt, dU3dt, dVdt]
        deriv_mat = np.column_stack((dUdt,dVdt))
        #print("deriv_mat",deriv_mat)
        deriv_vec = deriv_mat.flatten()
        #print("deriv_vec",deriv_vec)
        return deriv_vec


    def dynamics_HO_without_diffusion(self, t, x): #, args):

        #higher order integration 
        #this argument t is here just because I needed it for the solve_ivp function to work

        #x is now a vector -> put back to a matrix
        mat = np.reshape(x, (self.N, 2))
            
        U,V = mat[:,0],mat[:,1]

        dUdt = self.c*(-U**3 + U - V) #- diff_U
        dVdt = self.c * self.b* (U -self.a*V)
        #dUdt = self.r*U*(1-U/self.k) - self.q*U*V/(self.W + U) #- self.Cmat[0,0] * self.L @ U - self.Cmat[0,1] * self.L @ V
        #dUdt = (U**2+6*U+1)/self.k- self.q*U*V/(W + U) - self.Cmat[0,0] * self.L @ U + self.Cmat[0,1] * self.L @ V
        #dVdt = self.eta*self.q*U*V/(self.W + U) - self.M*V**2 #- self.Cmat[1,1]*self.L @ V - self.Cmat[1,0]*self.L @ U

        #flatten back so that I get a list [dU1dt, dV1dt, dU2dt, dVdt, dU3dt, dVdt]
        deriv_mat = np.column_stack((dUdt,dVdt))
        #print("deriv_mat",deriv_mat)
        deriv_vec = deriv_mat.flatten()
        #print("deriv_vec",deriv_vec)
        return deriv_vec
        
    


    def initial_conditions(self):
        
        #initial conditions
        
        mat = np.zeros((self.N, 2))

        if self.initial_mat is not None:

            mat = self.initial_mat
        
        elif self.initial == "uniform":

            col1 = np.asarray([self.U_s] * self.N)
            col2 = np.asarray([self.V_s] * self.N)

            mat = np.column_stack((col1,col2))
            #mat = mat + self.noisemat

        elif self.initial == "random":
    
            sampleU = np.random.uniform(low=np.nextafter(0.0, 1.0), high=1.0, size=self.N) #so that 0 is exluded, starts from the smallest value larger than 0
            sampleV = np.random.uniform(low=np.nextafter(0.0, 1.0), high=1.0, size=self.N)
            for i in range(self.N):
                mat[i,0] = sampleU[i]
                mat[i,1] = sampleV[i]
                
                
        elif self.initial == "half_random":
            
            noise_amplitude = 0.1 #can be whatever, can deviate from the base_value this much to either direction
            base_value = 1.5 #this as well
            
            sampleU = np.random.uniform(low=-noise_amplitude, high=noise_amplitude, size=self.N)
            sampleV = np.random.uniform(low=-noise_amplitude, high=noise_amplitude, size=self.N)
            for i in range(self.N):
                mat[i,0] = base_value + sampleU[i]
                mat[i,1] = base_value + sampleV[i]
                
        self.initial_mat = mat



    def check_turing(self, node):
    
        """ designed to bring the system to Turing
            slowly brings the system towards Turing where there starts to be spatial variation
            so we need to check if a node's neighborhood is similar to itself

            If either the density U or the density V is different (so two criterias to check currently)

            So just slowly change the parameter for all? After every Euler step? This is probably too much. And then decisively 
            change the parameter to the opposite direction if the neighborhood is too different. 
          
            - so I could calculate the average fractions of a node's neighbors
            Even though I guess that this kind of rule can lead to this kind of phenomenon where some parts of the network are different
            than the others but this change is so gradual that locally this is not observed.
            Or I could do a less local rule where I compare a node's values to the average of all nodes
            - I guess I could do other versions as well; such as if the diff even with one neighbor is over epsilon, then do something

            should I do this somehow using fractions or not """

        hit = False #hit == False means that similar enough
        
        #frac_own = self.U / self.V #concentrations of prey and predator
        U_own = self.U_means[node] #self.mat[node,0]
        V_own = self.V_means[node] #self.mat[node,1]

        #get the neighbors
        neighs = self.G[node]
        n_neighs = len(neighs)

        neighs_U,neighs_V = 0,0
        for neigh in neighs:
            U_neigh = self.U_means[neigh] #self.mat[neigh,0]
            V_neigh = self.V_means[neigh] #self.mat[neigh,1]
            neighs_U += U_neigh
            neighs_V += V_neigh

        U_avg_neigh = neighs_U / n_neighs
        V_avg_neigh = neighs_V / n_neighs

        if abs(U_avg_neigh - U_own) > self.turing_threshold or abs(V_avg_neigh - V_own) > self.turing_threshold:
            hit = True

        return hit
        



    
    def check_turing_all_nodes(self):
        
        """ applies check_turing to all nodes, so that all values can be updated after one time iteration,.
        
        arr is an arr of the form [False, True, True ...] where False if node similar enough to its neighbors (or 
        does not have neighbors), True if the threshold condition is surpassed 
        
        arr_over_too_different is an array of indices (node numbers as well) for which the threshold condition is surpassed,
        from here I can e.g. choose randomly which node to update """

        tval = self.res.t[-1] - 200 #so this will be i.e. 4800
        t_tail = 0 #index so that tvals[t_tail:] is the tail that we want to analyze
        for val in self.res.t:
            if val > tval:
                break
            t_tail += 1

        self.U_means = []
        self.V_means = []
        
        for node in self.nodes:

            uvals = self.res.y[int(node*2),t_tail:]
            vvals = self.res.y[int(node*2+1),t_tail:]
            U_own = np.mean(uvals)
            V_own = np.mean(vvals)
            self.U_means.append(U_own)
            self.V_means.append(V_own)

        arr = list(map(self.check_turing, self.nodes))
        arr_over_too_different = np.where(arr)[0]

        self.stdU = np.std(self.U_means)
        self.stdV = np.std(self.V_means)

        return arr_over_too_different

    

    def check_hopf_HO(self,node):

            
        uvals = self.res.y[int(node*2),self.t_tail_hopf:]
        vvals = self.res.y[int(node*2+1),self.t_tail_hopf:]
        amplitude_U = abs(max(uvals) - min(uvals))
        amplitude_V = abs(max(vvals) - min(vvals))
        if amplitude_U > self.max_amplitude_U:
            self.max_amplitude_U = amplitude_U
        if amplitude_V > self.max_amplitude_V:
            self.max_amplitude_V = amplitude_V

        hit = False
        
        if amplitude_U > self.hopf_threshold or amplitude_V > self.hopf_threshold:
            hit = True

        return hit




    def check_hopf_all_nodes(self):

        tval = self.res.t[-1] - 200
        t_tail = 0
        for val in self.res.t:
            if val > tval:
                break
            t_tail += 1

        self.t_tail_hopf = t_tail
        self.max_amplitude_U = 0
        self.max_amplitude_V = 0
        arr = list(map(self.check_hopf_HO, self.nodes))
        self.max_amplitudes_U.append(self.max_amplitude_U)
        self.max_amplitudes_V.append(self.max_amplitude_V)

        arr_over_too_different = np.where(arr)[0]

        return arr_over_too_different



    def turing_super_update_all_reverse(self,t):

        #chosen = np.random.randint(0,len(self.nodes), size =int( 0.5*len(self.nodes)))
        probs = np.random.uniform(low=0.0, high=1.0, size=len(self.nodes))
        chosen = np.where(probs < 0.5)[0]

        vals,_ = eigenvalues_of_laplacian(self.G)
        kappamax1 = max(vals)
        kn1 = sorted(vals)[1]

        n_restrict = 0
        
        for node in chosen:

            restrict = 0
            #weights = []
            #for n2 in self.G[node]:
                #print(node,n2)
                #weights.append(self.G[node][n2]["weight"])
            ws = [self.F[node][neigh]["weight"] for neigh in self.F[node]] #take the weights from before any structural updates
            median = np.median(ws)
            #val = max(self.wdict[node]) + self.pad
            #val = max(min(ws) - self.pad, self.min_link_weight-self.pad)
            
            for n2 in self.G[node]:
                w_old = self.F[node][n2]["weight"]
                w_current = self.G[node][n2]["weight"]
                if w_old <= median:
                    #if w == val:
                        #new_weight = max(w, self.min_link_weight) #min_link_weight
                    #else:

                    #new_weight = max(w_current - (np.real(w_old-val)**(-0.5)*self.remov_turing), self.min_link_weight)
                    new_weight = max(w_current - self.remov_turing, self.min_link_weight)

                    if new_weight <= self.min_link_weight:
                        restrict = 1
                    
                    #new_weight = max(w - (val-w)*self.remov_turing, self.min_link_weight)
                    self.G[node][n2]["weight"] = new_weight
                    diff = new_weight-w_current
    
                    if self.edgetrack == True:  
                        self.modification_ts.append(t)
                        self.modifications.append(diff) #-self.remov_turing)
                        self.mod_edges.append((node,n2))


            if restrict == 1:
                n_restrict += 1

                
        self.L = laplacian(self.G)

        vals,_ = eigenvalues_of_laplacian(self.G)
        kappamax2 = max(vals)
        kn2 = sorted(vals)[1]
        kappadiff = kappamax2 - kappamax1
        kappadiff2 = kn2-kn1
        self.effects_turing_super.append(kappadiff)
        self.effects_turing_super_kn.append(kappadiff2)
        self.fracs_tsup_restrict.append(n_restrict/len(chosen))



    
    def turing_super_update_all(self,t):

        #in this version super pushes eigenvalue up
        #chosen = np.random.randint(0,len(self.nodes), size =int( 0.5*len(self.nodes)))
        probs = np.random.uniform(low=0.0, high=1.0, size=len(self.nodes))
        chosen = np.where(probs < 0.5)[0]

        vals,_ = eigenvalues_of_laplacian(self.G)
        kappamax1 = max(vals)
        kn1 = sorted(vals)[1]

        for node in chosen:
            for n2 in self.G[node]:
                self.G[node][n2]["weight"] += self.increment_turing

                if self.edgetrack == True:
                    self.modification_ts.append(t)
                    self.modifications.append(self.increment_turing)
                    self.mod_edges.append((node,n2))

        self.L = laplacian(self.G)

        vals,_ = eigenvalues_of_laplacian(self.G)
        kappamax2 = max(vals)
        kn2 = sorted(vals)[1]
        kappadiff = kappamax2 - kappamax1
        kappadiff2 = kn2-kn1
        self.effects_turing_super.append(kappadiff)
        self.effects_turing_super_kn.append(kappadiff2)
        

    
    def hopf_super_update_all(self, t):

        #chosen = np.random.randint(0,len(self.nodes), size =int( 0.5*len(self.nodes)))
        probs = np.random.uniform(low=0.0, high=1.0, size=len(self.nodes))
        chosen = np.where(probs < 0.5)[0]

        vals,_ = eigenvalues_of_laplacian(self.G)
        kappamax1 = max(vals)
        kn1 = sorted(vals)[1]
        
        for node in chosen:
            
            #weights = []
            
            ws = [self.F[node][neigh]["weight"] for neigh in self.F[node]]
            ws = np.real(ws)
            median = np.median(ws)
            """
            val = max(ws) + self.pad
            #for n2 in self.G[node]:
                #print(node,n2)
                #weights.append(self.G[node][n2]["weight"])
            #val = max(min(ws) - self.pad, self.min_link_weight-self.pad)
            """
            for n2 in self.G[node]:
                w_old = self.F[node][n2]["weight"]
                w_current = self.G[node][n2]["weight"]
                if w_old >= median:
                    #if w_old == median:
                        #new_weight = w_current + (w_old-median)*self.increment_hopf
                    #else:
                    #new_weight = w_current + (val-w_old)**(0.5)*self.increment_hopf #, self.min_link_weight)
                    new_weight = w_current + self.increment_hopf 
                    #new_weight = self.G[node][n2]["weight"] + (w-val)*self.increment_hopf
                    self.G[node][n2]["weight"] = new_weight
                    diff = new_weight-w_current
    
                    if self.edgetrack == True:
                        self.modification_ts.append(t)
                        self.modifications.append(diff) #-self.remov_turing)
                        self.mod_edges.append((node,n2))


        #for n1,n2,d in self.G.edges(data=True):
            #self.G[n1][n2]['weight'] += self.increment_hopf

        self.L = laplacian(self.G)

        vals,_ = eigenvalues_of_laplacian(self.G)
        kappamax2 = max(vals)
        kn2 = sorted(vals)[1]
        kappadiff = kappamax2 - kappamax1
        self.effects_hopf_super.append(kappadiff)
        kappadiff_kn = kn2-kn1
        self.effects_hopf_super_kn.append(kappadiff_kn)
        


    def turing_sub_update_reverse(self, node, t):

        neighs = list(self.G[node])
        n_neighs = len(neighs)

        ws = [self.F[node][neigh]["weight"] for neigh in self.F[node]]
        ws = np.real(ws)
        median = np.median(ws)
        #val = max(self.wdict[node]) + self.pad
        #val = max(min(ws) - self.pad, self.min_link_weight-self.pad)
        for neigh in neighs:
            w_old = self.F[node][neigh]["weight"]
            w_current = self.G[node][neigh]["weight"]
            if w_old <= median:
                #new_weight = max(self.G[node][n2]["weight"] -  (val-w)*self.remov_turing, self.min_link_weight)
                #new_weight = w + (val-w)*self.increment_turing
                #new_weight = w_current + (w_old-val)**(-1)*self.increment_turing
                
                #new_weight = w_current + (w_old-val)**(-0.5)*self.increment_turing
                new_weight = w_current + self.increment_turing
                
                self.G[node][neigh]["weight"] = new_weight
                diff = new_weight-w_current
    
                if self.edgetrack:
                    self.modification_ts.append(t)
                    self.modifications.append(diff) 
                    self.mod_edges.append((node,neigh))

        
        #update Laplacian
        self.L = laplacian(self.G)

    
    def turing_sub_update(self, node, t):

        neighs = list(self.G[node])
        n_neighs = len(neighs)

        for neigh in neighs:
            old_weight = self.G[node][neigh]['weight'] 
            new_weight = max(self.G[node][neigh]['weight']-self.remov_turing, self.min_link_weight)
            self.G[node][neigh]['weight'] = new_weight
            diff = new_weight - old_weight

            if self.edgetrack == True:
                self.modification_ts.append(t)
                self.modifications.append(diff)
                self.mod_edges.append((node,neigh))
                #print("mod:", node,neigh)
                #print("   neigh", str(neigh), self.G[node][neigh]['weight'])
            
        #update Laplacian
        
        self.L = laplacian(self.G)
    

        

    def hopf_sub_update(self, node, t):

        neighs = list(self.G[node])
        n_neighs = len(neighs)
        
        #weights = []
        #for neigh in neighs:
            #weights.append(self.G[node][neigh]["weight"])
        #val = max(min(self.wdict[node]) - self.pad, self.min_link_weight)
        
        ws = [self.F[node][neigh]["weight"] for neigh in self.F[node]]
        median = np.median(np.real(ws))
        #val = max(ws) + self.pad
        
        restrict = 0
        
        for neigh in neighs:
            w_old = self.F[node][neigh]['weight'] 
            w_current = self.G[node][neigh]['weight'] 
            if w_old >= median:
                #w = old_weight - (old_weight-val)*self.remov_hopf
                #new_weight = max(w, self.min_link_weight)
                #new_weight = max(w_current - (val-w_old)**(0.5)*self.remov_hopf, self.min_link_weight)
                new_weight = max(w_current - self.remov_hopf, self.min_link_weight)
                if w_current - self.remov_hopf < self.min_link_weight:
                    restrict = 1
                self.G[node][neigh]['weight'] = new_weight
                diff = new_weight - w_current
    
                if self.edgetrack == True:
                    self.modification_ts.append(t)
                    self.modifications.append(diff)
                    self.mod_edges.append((node,neigh))
                    #print("mod:", node,neigh)
                    #print("   neigh", str(neigh), self.G[node][neigh]['weight'])

        #update Laplacian
        self.L = laplacian(self.G)

        return restrict

        
        
    def simulate(self):

        #these are just for plotting purposes, will be used when recording != None
        self.tvals = []
        self.Uvals = []
        self.Vvals = []
        self.tvals2 = []
        self.Uvals2 = []
        self.Vvals2 = []
        self.tvals3 = []
        self.Uvals3 = []
        self.Vvals3 = []
        
        self.initial_conditions() 
        #the above creates self.initial_mat for the values of U and V (array N times 2)
        
        vec = self.initial_mat.flatten() #flatten to form [U_1,V_1,U_2,V_2,U_3,V_3] 

        print("nsteps",self.n_tsteps, "tstep", self.tstep)
        firstwrite = True #just technical thing to writeout stuff
        firstwrite_edgechange = True
        count = 0
        true_t = 0 

        if self.edgetrack == True:
            #write out graph
            edgechangesfile_tstart = "../files/edgechanges/" + self.graphname + "_tstart_"+str(true_t)
            nx.write_weighted_edgelist(self.G, edgechangesfile_tstart +".edgelist")

        
        for tt in range(self.n_tsteps): #this tstep thing might need to be changed?


            if self.investigate_divergence == True and self.final_reached == None:
                #this is not working currently
                diffmat = self.mat - self.final_mat
                diff_val = np.abs(np.sum(diffmat[:,0]) + np.sum(diffmat[:,1]))
                if diff_val < 0.0001:
                    self.final_reached = true_t

            #add noise
            #noise1 = np.random.uniform(low=-self.noise_amplitude, high=self.noise_amplitude, size=self.N)
            #noise2 = np.random.uniform(low=-self.noise_amplitude, high=self.noise_amplitude, size=self.N)
            #noise = np.column_stack((noise1,noise2))
            #self.mat = self.mat + noise
            #add noise to the flattened vec
            noise1 = np.random.uniform(low=-self.noise_amplitude, high=self.noise_amplitude, size=self.N*2)
            vec = vec + noise1

            if self.diffusion_on == True:
                res = solve_ivp(self.dynamics_HO, (true_t,true_t+self.tstep), vec, method = "RK45") #method = "RK45" #or: Radau 
                #always integrates exactly untic tstep
                #changed so that directly integrates from true_t
            else:
                res = solve_ivp(self.dynamics_HO_without_diffusion, (true_t,true_t+self.tstep), vec, method = "RK45") #method = "RK45" #or: Radau

            if res.status != 0: #i.e. it failed, should not occur at all
                print("integration failed!!!, interrupting")
                return

            print(sys.getsizeof(res))
            self.res = res
            vec = res.y[:,-1] #taking the results at the end
            #note that res.y gives a matrix of dimensions vec x timepoints
            
            #res = Radau(self.dynamics, 0, vec, self.tstep)
            #self.mat = np.reshape(vec, (self.N, 2))
            
            if true_t == 0: #this is to adjust for repeating end/start values in subsequent integration rounds
                adjust_ind = 0
            else:
                adjust_ind = 1
                
            self.tvals += list(res.t)[adjust_ind:] # leaves the repeating first value out

            if self.recording != None:
                self.Uvals += list(res.y[self.recording[0]*2,:][adjust_ind:]) #res.y[0])
                self.Vvals += list(res.y[self.recording[0]*2+1,:][adjust_ind:]) #res.y[1])
                self.Uvals2 += list(res.y[self.recording[1]*2,:][adjust_ind:]) #res.y[0])
                self.Vvals2 += list(res.y[self.recording[1]*2+1,:][adjust_ind:]) 
                self.Uvals3 += list(res.y[self.recording[2]*2,:][adjust_ind:]) #res.y[0])
                self.Vvals3 += list(res.y[self.recording[2]*2+1,:][adjust_ind:])

            true_t += self.tstep
            #recorded tvals will be [tstep,2*tstep ...]
            #other vals [val at interval 0-tstep, val at interval tstep-2*tstep ...]
            
            #TOPOLOGY UPDATES AFTER INTEGRATION
            
            #so this following if-stuff will be done if either or both of hopf/turing-updates are on
            if self.turing_update == True or self.hopf_update == True: # and tt != 0:

                #this is all the tracking stuff
                eig_di = nx.eigenvector_centrality(self.G,weight = "weight")
                eig_di_unweighted = nx.eigenvector_centrality(self.G)
                self.F = self.G.copy() #this stores the old values before structural updates

                avg_deg = np.trace(self.L)/self.N
                self.avg_degs.append(avg_deg)
                weights = []
                for _, _, data in self.G.edges(data = 'weight'):
                    weights.append(data)
                weight_std = np.std(weights)
                #weight_std = round(weight_std, 4)
                self.weight_stds.append(weight_std)
                
                eigvals,eigvecs = np.linalg.eig(self.L)
                sorted_eigvals = sorted(eigvals)
                kn = sorted_eigvals[1]
                lam = sorted_eigvals[-1]
                self.lambdas.append(lam)
                self.secondsmallest.append(kn)
                self.tracking_tvals.append(true_t)
                print(true_t, lam)

                mean_eigcent = np.mean(list(eig_di.values()))
                std_eigcent = np.std(list(eig_di.values()))
                self.eigcents_all.append(mean_eigcent)
                self.eigcent_stds.append(std_eigcent)
                violating_nodes_turing = self.check_turing_all_nodes()
                eigcents = []
                eigcents_unweighted = []
                for node in violating_nodes_turing:
                    eigcents.append(eig_di[node])
                    eigcents_unweighted.append(eig_di_unweighted[node])
                    
                self.eigcents_turing.append(np.mean(eigcents))
                self.eigcents_turing_unweighted.append(np.mean(eigcents_unweighted))
                self.turing_stdUs.append(self.stdU)
                self.turing_stdVs.append(self.stdV)
                self.violating_turing = violating_nodes_turing
                self.n_violating_turing.append(len(violating_nodes_turing))


                #then the Hopf checks
                violating_nodes_hopf = self.check_hopf_all_nodes()
                eigcents = []
                eigcents_unweighted = []
                
                for node in violating_nodes_hopf:
                    eigcents.append(eig_di[node])
                    eigcents_unweighted.append(eig_di_unweighted[node])

                self.eigcents_hopf.append(np.mean(eigcents))
                self.eigcents_hopf_unweighted.append(np.mean(eigcents_unweighted))
                self.violating_hopf = violating_nodes_hopf
                self.n_violating_hopf.append(len(violating_nodes_hopf))
                self.overlaps.append(len(set(self.violating_turing) & set(self.violating_hopf)))

                count += 1

                #TURING SUBCRITICALITY UPDATE
                if self.turing_update == True: #do the turing sub update
                    
                    kappamax1 = lam #max(vals)
                    kn1 = kn
        
                    if len(violating_nodes_turing) > 0:

                        print("n violating",len(violating_nodes_turing)) #, violating_nodes)
    
                        for node_to_update in violating_nodes_turing:
                            if self.reverse_turing == False:
                                self.turing_sub_update(node_to_update,true_t)
                            else:
                                self.turing_sub_update_reverse(node_to_update,true_t)
                            
                    else:
                        print("no violating")
                
                    vals,_ = eigenvalues_of_laplacian(self.G)
                    kappamax2 = max(vals)
                    kn2 = sorted(vals)[1]
                    kappadiff = kappamax2 - kappamax1
                    kappadiff2 = kn2-kn1
                    self.effects_turing_sub.append(kappadiff)
                    self.effects_turing_sub_kn.append(kappadiff2)
                else:
                    self.effects_turing_sub.append(0)
                    self.effects_turing_sub_kn.append(0)


                #TURING SUPERCRITICALITY UPDATE
                
                if count == self.count and self.turing_update==True:
                    
                    if self.reverse_turing == False:
                        self.turing_super_update_all(true_t) #increase weights of all links
                    else:
                        self.turing_super_update_all_reverse(true_t)
                    
                    if self.hopf_update == False:
                        count = 0
                else:
                    self.effects_turing_super.append(0)
                    self.effects_turing_super_kn.append(0)
                    self.fracs_tsup_restrict.append("no")

            
                #HOPF SUBCRITICALITY UPDATE
                if self.hopf_update == True:
                    
                    vals,_ = eigenvalues_of_laplacian(self.G)
                    kappamax1 = max(vals)
                    kn1 = sorted(vals)[1]
        
                    if len(violating_nodes_hopf) > 0:
                        print("n violating",len(violating_nodes_hopf))

                        n_restrict_hopf = 0
                        for node_to_update in violating_nodes_hopf:
                            restrict = self.hopf_sub_update(node_to_update,true_t)
                            if restrict == 1:
                                n_restrict_hopf += 1
                        self.fracs_hsub_restrict.append(n_restrict_hopf/len(violating_nodes_hopf))
    
                    else:
                        print("no violating")
                        self.fracs_hsub_restrict.append("no")

                    vals,_ = eigenvalues_of_laplacian(self.G)
                    kappamax2 = max(vals)
                    kn2 = sorted(vals)[1]
                    kappadiff = kappamax2 - kappamax1
                    kappadiff2 = kn2-kn1
                    self.effects_hopf_sub.append(kappadiff)
                    self.effects_hopf_sub_kn.append(kappadiff2)
                else:
                    self.effects_hopf_sub.append(0)
                    self.effects_hopf_sub_kn.append(0)

                #HOPF SUPERCRITICALITY UPDATE
                if count == self.count and self.hopf_update==True:
                    self.hopf_super_update_all(true_t) #this is for now the same as for Turing
                    count = 0
                else:
                    self.effects_hopf_super.append(0)
                    self.effects_hopf_super_kn.append(0)


            #writeout_step = 500/self.tstep
            #print("lenn:", len(self.tracking_tvals),self.tracking_tvals)
            if self.writeout == True and len(self.tracking_tvals) == 10: #tt % writeout_step == 0: 
                print("writing out")
                #print("eigcents_turing",eigcents_turing)
                timeseriesfile = "../files/timeseries/" + self.graphname + ".csv"
                edgechangesfile = "../files/edgechanges/" + self.graphname + ".csv"
                statesfile = "../files/states/" + self.graphname + ".csv"
                if firstwrite == True: 
                    write_several_rows(timeseriesfile,[self.tracking_tvals, self.avg_degs, self.lambdas, self.n_violating_turing,self.n_violating_hopf,self.secondsmallest, self.weight_stds,self.overlaps,self.turing_stdUs, self.turing_stdVs, self.max_amplitudes_U, self.max_amplitudes_V, self.eigcents_all, self.eigcent_stds, self.eigcents_hopf, self.eigcents_turing,self.eigcents_hopf_unweighted, self.eigcents_turing_unweighted , self.effects_hopf_sub,self.effects_hopf_super,self.effects_turing_sub,self.effects_turing_super,self.effects_hopf_sub_kn,self.effects_hopf_super_kn,self.effects_turing_sub_kn,self.effects_turing_super_kn,self.fracs_tsup_restrict, self.fracs_hsub_restrict],"w",rounding = True)
                    firstwrite = False
                else:
                    write_several_rows(timeseriesfile,[self.tracking_tvals, self.avg_degs, self.lambdas ,self.n_violating_turing,self.n_violating_hopf,self.secondsmallest, self.weight_stds,self.overlaps,self.turing_stdUs, self.turing_stdVs, self.max_amplitudes_U, self.max_amplitudes_V, self.eigcents_all, self.eigcent_stds, self.eigcents_hopf, self.eigcents_turing,self.eigcents_hopf_unweighted, self.eigcents_turing_unweighted , self.effects_hopf_sub,self.effects_hopf_super,self.effects_turing_sub,self.effects_turing_super,self.effects_hopf_sub_kn,self.effects_hopf_super_kn,self.effects_turing_sub_kn,self.effects_turing_super_kn,self.fracs_tsup_restrict, self.fracs_hsub_restrict],"a",rounding = True)
                    
                if self.edgetrack == True:
                    if firstwrite_edgechange ==True:
                        write_several_rows(edgechangesfile, [self.modification_ts,self.modifications,self.mod_edges],"w")  
                        firstwrite_edgechange = False
                    else:
                        write_several_rows(edgechangesfile, [self.modification_ts,self.modifications,self.mod_edges],"a")
                #print("tracking_tvals",self.tracking_tvals)
                self.tracking_tvals = []
                self.avg_degs = []
                self.lambdas = []
                self.secondsmallest = []
                self.n_violating_turing = []
                self.n_violating_hopf = []
                self.weight_stds = []
                self.overlaps = []
                self.turing_stdUs = []
                self.turing_stdVs = []
                self.max_amplitudes_U = []
                self.max_amplitudes_V = []
                self.eigcents_all = []
                self.eigcent_stds = []
                self.eigcents_hopf = []
                self.eigcents_turing = []
                self.eigcents_hopf_unweighted = []
                self.eigcents_turing_unweighted = []

                self.effects_hopf_sub = []
                self.effects_hopf_super = []
                self.effects_turing_sub = []
                self.effects_turing_super = []

                self.effects_hopf_sub_kn = []
                self.effects_hopf_super_kn = []
                self.effects_turing_sub_kn = []
                self.effects_turing_super_kn = []

                self.modification_ts = []
                self.modifications = []
                self.mod_edges = []
                self.fracs_tsup_restrict = []
                self.fracs_hsub_restrict = []
                
                
                #write the states (always overwrites)
                write_several_rows(statesfile, [vec],"w")

        
        #writeout the last remaining if there are any
        if len(self.tracking_tvals) != 0:
            #print("tracking_tvals here",self.tracking_tvals)
            #print("avg_degs here",self.avg_degs)
            timeseriesfile = "../files/timeseries/" + self.graphname + ".csv"
            write_several_rows(timeseriesfile,[self.tracking_tvals, self.avg_degs, self.lambdas,self.n_violating_turing,self.n_violating_hopf,self.secondsmallest, self.weight_stds,self.overlaps,self.turing_stdUs, self.turing_stdVs, self.max_amplitudes_U, self.max_amplitudes_V, self.eigcents_all, self.eigcent_stds, self.eigcents_hopf, self.eigcents_turing,self.eigcents_hopf_unweighted, self.eigcents_turing_unweighted , self.effects_hopf_sub,self.effects_hopf_super,self.effects_turing_sub,self.effects_turing_super,self.effects_hopf_sub_kn,self.effects_hopf_super_kn,self.effects_turing_sub_kn,self.effects_turing_super_kn,self.fracs_tsup_restrict, self.fracs_hsub_restrict],"a",rounding = True)
            if self.edgetrack == True:
                write_several_rows(edgechangesfile, [self.modification_ts,self.modifications,self.mod_edges],"a")
        


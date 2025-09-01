# +
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import math
import csv




def read_graph_correct(folder,name):

    """ this needs to be build this way to ensure that the Laplacian will be in a form that you would expect """
    F = nx.read_weighted_edgelist(folder + name +".edgelist",nodetype=int)
    nodes = sorted(F.nodes())
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for n1,n2,d in F.edges(data=True):
        w = d["weight"]
        G.add_edge(n1,n2,weight=w)

    return G
    


def write_line(outfile,line,mode="w",rounding=False):
    
    if rounding == True:
        line2 = []
        for val in line:
            if type(val) == str:
                line2.append(val)
            else:
                line2.append(round(val,8))
        line = line2
             
    with open(outfile, mode) as csvFile:
        writer = csv.writer(csvFile, delimiter=' ')
        writer.writerow(line)



def write_several_rows(outfile,list_of_lists,mode='w',rounding = False):

    """for example [[1,2,3],[4,5,6],[7,8,9]] is written so that each of the smaller lists becomes one column """

    #take the length of the longest lis
    lislen = 0
    jj = 0
    for lis in list_of_lists:
        l = len(lis)
        if l > lislen:
            lislen = l
            #print("winning",jj)
        jj += 1

    #print("lislen",lislen)
    #print(type(list_of_lists[0][0]))
    
    with open(outfile, mode) as csvFile:
        writer = csv.writer(csvFile, delimiter=' ')
        
        n_iterates = len(list_of_lists)
        #lislen = len(list_of_lists[0])
        
        for varind in range(lislen):
            line = []
            for i in range(n_iterates):
                lis = list_of_lists[i]
                if varind < len(lis):
                    val = lis[varind]
                    if rounding == True:
                        if type(val) == str:
                            line.append(val)
                        else:
                            line.append(round(val,8))
                    else:
                        line.append(val)
                else:
                    line.append("missing")
            writer.writerow(line)


def ER_graph_undirected(avg_deg = 2.0,n = 10**4):
    
    """ creates an ER-graph with avg_deg*n links """
    
    n_links = int(int(avg_deg * n)/2) #because this is undirected, needs to be divided
    
    G = nx.DiGraph()
    nodes = np.arange(0,n)
    rans = np.random.randint(0,n,5*n_links +500)
    
    G.add_nodes_from(nodes)
    
    i=hits=0
    
    while hits < n_links:
        n1,n2 = rans[i],rans[i+1]
        if not n1 == n2 and G.has_edge(n1,n2) == False and G.has_edge(n2,n1)==False:
            hits += 1
            G.add_edge(n1,n2, weight = 1.)
            #if unidirec == True:
            #if not G.has_edge(n2,n1):
                #G.add_edge(n2,n1)
        i += 2
        
    G = G.to_undirected()

    return G,n





#G,n = ER_graph_undirected()
#G.number_of_edges()/G.number_of_nodes()


# -
def ER_graph(avg_deg = 2.0, n = 10**4):
    
    """ creates an ER-graph with avg_deg*n links """
    
    n_links = int(avg_deg * n)
    
    G = nx.DiGraph()
    nodes = np.arange(0,n)
    rans = np.random.randint(0,n,5*n_links +500)
    
    G.add_nodes_from(nodes)
    
    i=hits=0
    while hits < n_links:
        n1,n2 = rans[i],rans[i+1]
        if not n1 == n2 and G.has_edge(n1,n2) == False:
            hits += 1
            G.add_edge(n1,n2)
            #if unidirec == True:
            #if not G.has_edge(n2,n1):
                #G.add_edge(n2,n1)
        i += 2

    return G,n



import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx
import scipy
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
import helpers
import importlib
import igraph
import datetime
import argparse

parser = argparse.ArgumentParser(description='Calculate edge betweenness in graph given in text file.')
parser.add_argument('--graph')
parser.add_argument('--k', type=int)
parser.add_argument('--alg')
args = parser.parse_args()

if args.graph is None:
    print("Need to provide graph!")
    exit(1)

if args.k is None:
    print("Need to provide k")
    exit(1)

if args.alg is None:
    print("Need to provide clustering alg!")
    exit(1)

print("Starting computation...")
start = datetime.datetime.utcnow()
print(start)
print("Graph file = {}, k = {}".format(args.graph, args.k))
# Load graph and adjacency matrix
graph_file = "{}.txt".format(args.graph)

graph_data, header = helpers.load_graph(graph_file)
adjacency_matrix = helpers.calculate_adjacency_matrix(graph_data)

ig_graph = igraph.Graph.Adjacency((adjacency_matrix > 0).tolist())
ig_graph.es['weight'] = adjacency_matrix[adjacency_matrix.nonzero()]
ig_graph.vs['label'] = list(range(len(adjacency_matrix)))
ig_graph = ig_graph.as_undirected()

if args.alg == 'between':
    clustering = ig_graph.community_edge_betweenness(clusters=args.k, directed=False)

elif args.alg == 'spinglass':
    clustering = ig_graph.community_spinglass(spins=args.k)

else:
    print("Invalid cluster alg given!")
    exit(1)

end = datetime.datetime.utcnow()
print(end)

try:
    clustering = clustering.as_clustering(n=args.k)
except:
    pass

cluster = clustering.membership



fn = "{}_{}_clusters.npy".format(args.graph, args.alg)
np.save(fn, cluster)

print("Elapsed: ")
print(end - start)
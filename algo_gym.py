import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx
import scipy
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
import helpers
import itertools




graph_files = [
        "ca-AstroPh.txt", "ca-CondMat.txt",
        "ca-GrQc.txt", "ca-HepPh.txt", "ca-HepTh.txt" ]
Ks = [2]

if __name__ == '__main__':

    assert(len(graph_files) == len(Ks))

    for graph_file, k in zip(graph_files, Ks):

        graph_data, header = helpers.load_graph(graph_file)
        adjacency_matrix = helpers.calculate_adjacency_matrix(graph_data)
        graph = nx.from_numpy_matrix(adjacency_matrix)

        cluster_algos = [
                (KMeans(n_clusters=k), 'KMeans'),
                (helpers.BalancedKMeans(k, graph_data, n_init=10), 'Balanced Kmeans')]

        L_rw = helpers.calculate_normalized_random_walk_laplacian(adjacency_matrix)

        # Laplacian norm with nx and its U_norm
        L_norm = nx.normalized_laplacian_matrix(graph)
        U_norm = helpers.calculate_U_norm(L_norm, k)

        representations = [(U_norm, 'U_norm'), (L_rw, 'normalized random walk laplacian')]
        algo_pairs = itertools.product(cluster_algos, representations)

        for algo, data in algo_pairs:

            labels = algo[0].fit_predict(data[0])
            loss = helpers.objective_function(graph_data, labels)
            print(algo[1], data[1], loss)

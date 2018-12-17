import numpy as np
import os
import networkx as nx
from sklearn.cluster import KMeans
import helpers
import itertools
import csv
import scipy

def write_result(labels, graph_file, header):

    save_path = graph_file[:-4] + ".output"
    with open(save_path, 'w') as f:
        f.write(header + "\n")
        for vertex_id, label in enumerate(labels):
            f.write(str(vertex_id) + " " + str(int(label)) + "\n")
        f.close()

def update_loss(loss, algo, representation, graph_file, k):
    row = {'loss': loss, 'algo': algo,
            'representation': representation,
            'file': graph_file, 'k': k}
    csv_path = CSV_LOG_PATH
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=row.keys())
        if not(file_exists):
            writer.writeheader()
        writer.writerow(row)
        f.close()

graph_files = ["ca-GrQc.txt"]
Ks = [2]

#graph_files = [
#        "ca-GrQc.txt", "Oregon-1.txt",
#        "roadNet-CA.txt", "soc-Epinions1.txt", "web-NotreDame.txt" ]
#
#Ks = [2, 5, 50, 10, 20]

CSV_LOG_PATH = 'log.csv'

if __name__ == '__main__':

    assert(len(graph_files) == len(Ks))

    for graph_file, k in zip(graph_files, Ks):

        graph_data, header = helpers.load_graph(graph_file)
        adjacency_matrix = helpers.calculate_adjacency_matrix(graph_data)
        graph = nx.from_numpy_matrix(adjacency_matrix)

        cluster_algos = [
                (KMeans(n_clusters=k), 'KMeans'),
                (helpers.BalancedKMeans(k, n_init=10, graph_data=graph_data), 'Balanced Kmeans')]

        # L_rw = helpers.calculate_normalized_random_walk_laplacian(adjacency_matrix)

        # Laplacian norm with nx and its U_norm
        L_norm, _ = scipy.sparse.csgraph.laplacian(adjacency_matrix, normed=True,
                                    return_diag=True)
        U_norm = helpers.calculate_U_norm(L_norm, k)

        representations = [(U_norm, 'U_norm')]
        algo_pairs = itertools.product(cluster_algos, representations)

        best_loss = np.inf
        best_labels = []

        for algo, data in algo_pairs:
            print("Running: {} {} {}".format(graph_file, algo[1], data[1]))
            labels = algo[0].fit_predict(data[0])
            loss = helpers.objective_function(graph_data, labels)

            if (loss < best_loss):
                best_labels = labels
                best_loss = loss
            print(algo[1], data[1], loss)
            update_loss(loss, algo[1], data[1], graph_file, k)

        write_result(best_labels, graph_file, header)

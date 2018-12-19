import numpy as np
import os
import networkx as nx
from sklearn.cluster import KMeans
import helpers
import itertools
import csv
import scipy
import multiprocessing as mp


def fit_algo(algo_pair):

    algo = algo_pair[0][0]
    data = algo_pair[1][0]
    algo_name = algo_pair[0][1]
    data_name = algo_pair[1][1]

    print("Running ", algo_name, data_name)

    labels = algo.fit_predict(data)

    return (labels, algo_name, data_name)

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

graph_files = ["ca-GrQc.txt", "Oregon-1.txt",
        "soc-Epinions1.txt", "web-NotreDame.txt" ]

Ks = [2, 5, 10, 20]

CSV_LOG_PATH = 'log.csv'

if __name__ == '__main__':

    assert(len(graph_files) == len(Ks))

    for graph_file, k in zip(graph_files, Ks):
        print("Running {}".format(graph_file))

        graph_data, header = helpers.load_graph(graph_file)
        print("Graph data loaded.")
        adjacency_matrix = helpers.calculate_adjacency_matrix(graph_data)

        cluster_algos = [
                (KMeans(n_clusters=k, n_init=30), 'KMeans'),
                (helpers.BalancedKMeans(k, n_init=30, graph_data=graph_data), 'Balanced Kmeans')]

        # L_rw = helpers.calculate_normalized_random_walk_laplacian(adjacency_matrix)

        # Laplacian norm with nx and its U_norm
        L_norm, dd = scipy.sparse.csgraph.laplacian(adjacency_matrix, normed=True,
                                    return_diag=True)
        U_norm = helpers.calculate_U_norm(L_norm, k)

        embedding = helpers.calculate_embedding_representation(L_norm, dd, k)

        representations = [(U_norm, 'U_norm'), (embedding, 'embedding')]
        algo_pairs = list(itertools.product(cluster_algos, representations))
        algo_pairs.append(((helpers.FastModularity(k, adjacency_matrix), 'Fast Modularity'), (None, "Nothing for fast modularity")))

        best_loss = np.inf
        best_labels = []

        print("Laplacian calculated. Starting clustering...")

        pool = mp.Pool(len(algo_pairs))
        results = [pool.apply(fit_algo, args=(algo_pairs[idx],)) for idx in range(len(algo_pairs))]

        best_loss = np.inf
        best_labels = []
        best_algo = " "

        for labels, algo_name, data_name in results:

            loss = helpers.objective_function(graph_data, labels)

            if (loss < best_loss):
                best_labels = labels
                best_loss = loss
                best_algo = algo_name
            print(algo_name, data_name, loss)
            update_loss(loss, algo_name, data_name, graph_file, k)

        print("Best algo for ", graph_file, best_algo)
        write_result(best_labels, graph_file, header)

        results = []

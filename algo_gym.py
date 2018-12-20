import numpy as np
import os
import networkx as nx
from sklearn.cluster import KMeans
import helpers
import itertools
import csv
import scipy
import multiprocessing as mp
import argparse






args = parser.parse_args()

def fit_algo(algo_pair):

    algo = algo_pair[0][0]
    data = algo_pair[1][0]
    algo_name = algo_pair[0][1]
    data_name = algo_pair[1][1]

    print("Running ", algo_name, data_name)
    labels = algo.fit_predict(data)
    print("End ", algo_name, data_name)
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


# graph_files = ["ca-HepTh.txt", "ca-HepPh.txt", "ca-CondMat.txt", "ca-AstroPh.txt"]
# Ks = [20, 25, 100, 50]


# graph_files = ["ca-GrQc.txt", "Oregon-1.txt",
#        "soc-Epinions1.txt", "web-NotreDame.txt"]

# Ks = [2, 5, 10, 20]

parser = argparse.ArgumentParser(description='Algo Gym')

parser.add_argument('--log-to', default='log.csv', type=str, help='Default log.csv')

parser.add_argument('--kmean-workers', default=4, type=int, help='The number of workers for Kmean')

parser.add_argument('--no-balanced', dest='run_balanced', action='store_false', help='No Balanced KMean')
parser.set_defaults(run_balanced=True)

parser.add_argument('--no-kmean', dest='run_kmean', action='store_false', help='No regular KMean')
parser.set_defaults(run_kmean=True)

parser.add_argument('--no-fast', dest='run_fast', action='store_false', help='No Fast modularity')
parser.set_defaults(run_fast=True)

parser.add_argument('--load-graph', dest='run_fast', action='store_false', help='No Fast modularity')
parser.set_defaults(run_fast=True)

parser.add_arguments()

args = parser.parse_args()

CSV_LOG_PATH = args.log_to

if __name__ == '__main__':

    assert(len(graph_files) == len(Ks))

    for graph_file, k in zip(graph_files, Ks):
        print("Running {} with k={}".format(graph_file, k))

        graph_data, header = helpers.load_graph(graph_file)
        print("Graph data loaded.")
        adjacency_matrix = helpers.calculate_adjacency_matrix(graph_data)

        algo_pairs = []

        if (args.run_kmean or args.run_balanced):
            L_norm, dd = scipy.sparse.csgraph.laplacian(adjacency_matrix, normed=True, return_diag=True)
            embedding = helpers.calculate_embedding_representation(L_norm, dd, k)

            print("Laplacian calculated.")
            if args.run_kmean:
                algo_pairs.append((((KMeans(n_clusters=k, n_init=50, n_jobs=args.kmean_workers), 'KMeans'), (embedding, 'embedding'))))

            if args.run_balanced:
                algo_pairs.append(((helpers.BalancedKMeans(k, n_init=30, graph_data=graph_data), 'Balanced Kmeans'), (embedding, 'embedding')))

        if (args.run_fast):
            algo_pairs.append(((helpers.FastModularity(k, adjacency_matrix), 'Fast Modularity'), (None, "Nothing for fast modularity")))

        print("Starting clustering...")

        results = []
        pool = mp.Pool(processes=len(algo_pairs))
        for res in pool.imap(fit_algo, algo_pairs, chunksize=1):
            results.append(res)

        best_loss = np.inf
        best_labels = []
        best_algo = "No algos"

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

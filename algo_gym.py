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

def update_loss(loss, algo, representation, graph_file, k, csv_path):
    row = {'loss': loss, 'algo': algo,
            'representation': representation,
            'file': graph_file, 'k': k}
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a') as f:
        writer = csv.DictWriter(f, delimiter=',', fieldnames=row.keys())
        if not(file_exists):
            writer.writeheader()
        writer.writerow(row)
        f.close()

def parse_arguments():

    parser = argparse.ArgumentParser(description='Algo Gym')

    parser.add_argument('--log-to', default='log.csv', type=str, help='Default log.csv')

    parser.add_argument('--kmean-workers', default=4, type=int, help='The number of workers for Kmean')

    parser.add_argument('--balanced', dest='run_balanced', action='store_true', help='Run a Balanced KMean')
    parser.set_defaults(run_balanced=False)

    parser.add_argument('--kmean', dest='run_kmean', action='store_true', help='Runs a KMean')
    parser.set_defaults(run_kmean=False)

    parser.add_argument('--fast', dest='run_fast', action='store_true', help='Runs a Fast modularity')
    parser.set_defaults(run_fast=False)

    parser.add_argument('--stop-optimal', dest='stop_optimal', action='store_true', help='Stops algorithms at optimal clustering rather than enforcing k')
    parser.set_defaults(stop_optimal=False)

    parser.add_argument('--deep', dest='run_deep', action='store_true', help='Runs a Deep Walk')
    parser.set_defaults(run_deep=False)

    parser.add_argument('--labelprop', dest='run_labelprop', action='store_true', help='Runs Labelpropagation')
    parser.set_defaults(run_labelprop=False)


    parser.add_argument('--num-clusters', default=-1, dest='num_clusters', type=int, help='Check the correspondance for the graph. Pairs are commented in algo_gym.py')

    parser.add_argument('--one-k', dest='one_k', action='store_true', help='Only run on the k specified in num_clusters')
    parser.set_defaults(run_labelprop=False)

    parser.add_argument('--load-graph', default='ca-GrQc.txt', type=str, help='graph file')

    parser.add_argument('--all-graphs', dest='load_all', action='store_true', help='This ignores Web-NotreDame, zachary, and roadnet')
    parser.set_defaults(load_all=False)

    parser.add_argument('--non-competitive', dest='non_competitive', action='store_true', help='Non competitive graphs (Project part 1 graphs)')
    parser.set_defaults(non_competitive=False)

    parser.add_argument('--competitive', dest='competitive', action='store_true', help='Competitive graphs (ignores Web-NotreDame, zachary, and roadnet)')
    parser.set_defaults(competitive=False)

    args = parser.parse_args()

    return args


# You need to define at least one algo and the number clusters to run this!
if __name__ == '__main__':


    # These values are from the project documentation
    non_competitive_files = ["ca-HepTh.txt", "ca-HepPh.txt", "ca-CondMat.txt", "ca-AstroPh.txt"]
    non_competitive_Ks = [20, 25, 100, 50]

    competitive_files = ["ca-GrQc.txt", "Oregon-1.txt",
           "soc-Epinions1.txt"]
    competitive_Ks = [2, 5, 10]

    args = parse_arguments()

    graph_files = [args.load_graph]

    Ks = [args.num_clusters]

    if args.load_all:
        graph_files = non_competitive_files + competitive_files
        if not args.one_k:
            Ks = non_competitive_Ks + competitive_Ks

    if args.non_competitive:
        graph_files = non_competitive_files
        if not args.one_k:
            Ks = non_competitive_Ks

    if args.competitive:
        graph_files = competitive_files
        if not args.one_k:
            Ks = competitive_Ks

    if args.num_clusters < 1 and not(args.non_competitive or args.competitive or args.load_all):
        raise ValueError('You need to define the number of clusters!')

    #assert(len(graph_files) == len(Ks))
    if len(Ks) == 1 and len(graph_files) > 1:
        Ks = Ks*len(graph_files)

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
            algo_pairs.append(((helpers.FastModularity(k, adjacency_matrix, stop_at_optimal=args.stop_optimal), 'Fast Modularity'), (None, "Nothing for fast modularity")))

        if args.run_deep:
            savefile = graph_file[:-4] + '.embedding'
            algo_pairs.append(((helpers.DeepWalk(k, adjacency_matrix, num_walks=10, len_walk=40, embedding_savefile=savefile), 'Deep walk'), (None, "Nothing for deep walk")))


        if args.run_labelprop:
            import sys
            sys.setrecursionlimit(100000)
            algo_pairs.append(((helpers.LabelPropagation(k, adjacency_matrix, stop_at_optimal=args.stop_optimal), 'Labelpropagation'), (None, "Nothing for labelprop")))


        print("Starting clustering...")

        if len(algo_pairs) < 1:
            raise ValueError('No algorithms introduced!')

        print(algo_pairs)

        best_loss = np.inf
        best_labels = []

        pool = mp.Pool(processes=len(algo_pairs))
        for labels, algo_name, data_name in pool.imap(fit_algo, algo_pairs, chunksize=1):
            print("{} finished".format(algo_name))
            loss = helpers.objective_function(graph_data, labels)

            if (loss < best_loss):
                best_labels = labels
                best_loss = loss
                best_algo = algo_name

            print("{} with {} result: {}".format(algo_name, data_name, loss))

            update_loss(loss, algo_name, data_name, graph_file, k, args.log_to)
            write_result(best_labels, algo_name + "-"  + graph_file, header)

        print("Best algo for ", graph_file, best_algo)
        write_result(best_labels, graph_file, header)

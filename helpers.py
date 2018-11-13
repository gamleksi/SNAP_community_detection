import networkx as nx
import scipy
from sklearn.cluster import KMeans
import numpy as np
import os
import random

###### Graph loading

def load_graph(graph_file, graph_dir = "./graphs_processed"):
    path = os.path.join(graph_dir, graph_file)

    graph_data = []

    with open(path, 'r') as f:
        header = f.readline().rstrip() # rstrip removes trailing newline
        for line in f:
            line_strip = line.rstrip()
            components = line_strip.split(' ')
            line_data = (int(components[0]), int(components[1]))
            graph_data.append(line_data)
            
    graph_data = np.asarray(graph_data)

    return graph_data, header


def calculate_adjacency_matrix(graph_data):
    vertices = np.unique(graph_data.flatten())

    N = len(vertices)

    adjacency_matrix = np.zeros((N,N))

    for pair in graph_data:
        i = pair[0]
        j = pair[1]
        adjacency_matrix[i,j] = adjacency_matrix[j,i] = 1

    return adjacency_matrix


####### Spectral clustering
def calculate_degree_mat(A):
    # Calculates degree matrix from adjacency matrix A
    D = np.zeros_like(A)
    for i in range(len(D)):
        D[i,i] = np.sum(A[i,:]) # Node degree (i.e. num of edges terminating at that node) is row sum.
        
    return D

def calculate_normalized_laplacian(A, D):
    # Calculates normalzied laplacian matrix
    with np.errstate(divide='ignore'):
        D_tmp = 1.0 / np.sqrt(D)
    
    D_tmp[np.isinf(D_tmp)] = 0

    N = len(A)
    I = np.eye(N,N)
    L_norm = I - np.dot(np.dot(D_tmp, A), D_tmp)
    
    return L_norm

def spectral_cluster(adjacency_matrix=None, graph=None, k=2, normalized=True, cluster_alg=KMeans, random_state=None):
    # Either input adjacency matrix or networkx graph. Calculations with adjacency matrix is slower.
    assert(adjacency_matrix is not None or graph is not None)

    if random_state is None:
        random_state = random.randint(0,10000)
        # Still returns random results since eigenvalues iteration starts from random state...

    if graph is not None:
        if normalized:
            L = nx.normalized_laplacian_matrix(graph) 
        else:
            L = nx.laplacian_matrix(graph)
    
    else:
        A = adjacency_matrix
        D = calculate_degree_mat(A)
        
        if normalized:
            L = calculate_normalized_laplacian(A,D)

        else:
            L = D - A

    # Calculate eigenvectors matrix U
    eig = scipy.sparse.linalg.eigs(L, k)
    
    assert(np.sum(np.imag(eig[1])) == 0) # Drop imaginary values but assert that imaginary part must be zero
    
    U = np.real(eig[1]) # Pick only real values

    # Normalize U
    row_sums = U.sum(axis=1)
    U_norm = U / row_sums[:, np.newaxis]

    # Do clustering for U_norm
    clf = cluster_alg(n_clusters=k)
    C_labels = clf.fit_predict(U_norm)
    
    return U_norm, C_labels # Return U_norm for cluster debugging


##### Objective function specified in project
def objective_function(graph_data, labels):
    nominator = 0
    for edge in graph_data:
        v1 = edge[0]
        v2 = edge[1]
        if v1 == v2:
            continue
            
        if labels[v1] != labels[v2]:
            nominator += 1
            
    denominator = np.min(np.unique(labels, return_counts=True)[1]) # Min number of occurences of labels
    
    res = nominator / denominator
    return res
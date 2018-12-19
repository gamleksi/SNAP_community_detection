import networkx as nx
import scipy
from sklearn.cluster import KMeans
import numpy as np
import os
import random
import heapq
from itertools import count
import sklearn

###### Graph loading
def load_graph(graph_file, graph_dir = "./graphs_processed"):
    path = os.path.join(graph_dir, graph_file)

    graph_data = []

    with open(path, 'r') as f:
        header = f.readline().rstrip() # rstrip removes trailing newline
        for line in f:
            line_strip = line.rstrip()
            components = line_strip.split(' ')
            line_data = (int(components[0]) - 1, int(components[1]) - 1)
            graph_data.append(line_data)

        f.close()

    graph_data = np.asarray(graph_data)

    return graph_data, header


def calculate_adjacency_matrix(graph_data):
    vertices = np.unique(graph_data.flatten())

    N = len(vertices)

    adjacency_matrix = scipy.sparse.lil_matrix((N,N))  #np.zeros((N,N))

    for pair in graph_data:
        i = pair[0]
        j = pair[1]
        adjacency_matrix[i,j] = adjacency_matrix[j,i] = 1

    adjacency_matrix = adjacency_matrix.tocsr()
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

def calculate_normalized_random_walk_laplacian(A):
    # Calculates normalzied laplacian matrix
    D = calculate_degree_mat(A)

    with np.errstate(divide='ignore'):
        D_tmp = 1.0 / np.sqrt(D)

    D_tmp[np.isinf(D_tmp)] = 0
    N = len(A)
    I = np.eye(N,N)
    L_norm = I - np.dot(D_tmp, A)
    return L_norm

def calculate_U_norm(L, k):
    # Calculate eigenvectors matrix U
    eig = scipy.sparse.linalg.eigs(L, k + 1) # TODO onks taa nyt parempi?

    assert(np.sum(np.imag(eig[1])) == 0) # Drop imaginary values but assert that imaginary part must be zero
    U = np.real(eig[1]) # Pick only real values
    U = U[:,1:] # TODO is this correct?
    # Normalize U
    row_sums = U.sum(axis=1)
    U_norm = U / row_sums[:, np.newaxis]
    return U_norm

def calculate_embedding_representation(L, dd, k):

    laplacian = L
    laplacian *= -1
    random_state = sklearn.utils.check_random_state(None)
    v0 = random_state.uniform(-1, 1, laplacian.shape[0])
    lambdas, diffusion_map = scipy.sparse.linalg.eigsh(laplacian, k=k,
                                    sigma=1.0, which='LM',
                                    tol=0.0, v0=v0)
    embedding = diffusion_map.T[k::-1]
    embedding = embedding / dd

    embedding = sklearn.utils.extmath._deterministic_vector_sign_flip(embedding)
    embedding = embedding[:k].T

    return embedding


def spectral_cluster(adjacency_matrix=None, manual_laplacian=False, k=2, normalized=True, cluster_alg=KMeans, random_state=None, graph_data=None):


    # Either input adjacency matrix or networkx graph. Calculations with adjacency matrix is slower.
    # assert(adjacency_matrix is not None or graph is not None) TODO old?

    if random_state is None:
        random_state = random.randint(0,10000)
        # Still returns random results since eigenvalues iteration starts from random state...

    if not manual_laplacian:
        L, dd = scipy.sparse.csgraph.laplacian(adjacency_matrix, normed=normalized,
                                    return_diag=True)
    else:
        A = adjacency_matrix
        D = calculate_degree_mat(A)

        if normalized:
            L = calculate_normalized_laplacian(A,D)
        else:
            L = D - A

    U_norm = calculate_U_norm(L, k)

    # Do clustering for U_norm
    if graph_data is None:
        clf = cluster_alg(n_clusters=k, n_init=10)
    else:
        clf = cluster_alg(n_clusters=k, n_init=10, graph_data=graph_data)

    """
    embedding = calculate_embedding_representation(L, dd, k)
    """

    C_labels = clf.fit_predict(U_norm)

    return C_labels # Return U_norm for cluster debugging


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


##### KMEANS++
def plus_plus_init(X):
    # Data shape k x num data
    num_data = X.shape[0]
    num_clusters = X.shape[1]

    centers = np.zeros((num_clusters, num_clusters))

    # Choose randomly the first centrer
    centers[0] = X[np.random.random_integers(0, num_data-1)]

    for cluster_idx in range(num_clusters - 1):

        # Computes the minimum square norm distances to current centers
        distances = np.array([np.min([np.linalg.norm(X[x_idx] - centers[i]) ** 2 for i in range(1 + cluster_idx)]) for x_idx in range(num_data)])

        distance_probs = distances / np.linalg.norm(distances)
        cdf = np.cumsum(distance_probs)
        r = random.random()
        next_center_idx = np.where(cdf >= r)[0][0]
        centers[cluster_idx] = X[next_center_idx]

    return centers

##### Balanced KMEANS
class BalancedKMeans(object):

    def __init__(self, n_clusters, n_init=1, graph_data=None):

        self.n_clusters = n_clusters
        self.n_init = n_init
        assert(self.n_init == 1 or self.n_init > 1 and graph_data is not None)
        self.graph_data = graph_data

    def closest_centroid(self, point, centroids):
        centroid_distances = np.linalg.norm(point - centroids, axis=1)
        min_idx = np.argmin(centroid_distances)
        return min_idx, centroid_distances[min_idx]

    def picK_new_closest_centroid(self, point, centroids, prev_c_idx):
        centroid_distances = np.linalg.norm(point - centroids, axis=1)
        prev_distance = centroid_distances[prev_c_idx]
        cond = centroid_distances[prev_distance < centroid_distances]
        if not(cond.shape[0] > 0):
            raise ValueError('This should not occur.')

        assert(cond.shape[0] > 0)
        new_distance = np.min(cond)
        new_idx = np.where(new_distance == centroid_distances)[0][0]
        return new_idx, new_distance

    def heapsort(self, points, centroids, counter):
        h = []
        for value in points:
            centroid_idx, distance = self.closest_centroid(value, centroids)
            heapq.heappush(h, (distance, centroid_idx, next(counter), value))
        return h

    def new_centroid(self, cluster):
        return np.mean(cluster, axis=0)

    def get_labels(self, points, clusters):

        labels = np.zeros(points.shape[0], np.int)
        for i, cluster in enumerate(clusters):
            for point in cluster:
                cond = np.sum(points == point, axis=1) == point.shape[0]
                labels[cond] = i
        return labels

    def fit_predict(self, points, iterations=1000, diff=0.001):

        labels = np.zeros((self.n_init, points.shape[0]))
        results = np.zeros(self.n_init)

        for init_idx in range(self.n_init):
            centroids = plus_plus_init(points)
            size_limit = int(np.ceil(points.shape[0] / self.n_clusters))
            for i in range(iterations): # runs clustering until the number of iterations or the centroid does not change (diff)
                tiebreaker = count() # hacky way to ensure each element in heap is unique.
                h = self.heapsort(points, centroids, tiebreaker)
                clusters = [[] for i in range(self.n_clusters)]
                while(h):
                    distance, centroid_idx, _, value = heapq.heappop(h)
                    if len(clusters[centroid_idx]) < size_limit + 1:
                        clusters[centroid_idx].append(value)
                    else:
                        new_idx, new_distance = self.picK_new_closest_centroid(value, centroids, centroid_idx)
                        heapq.heappush(h, (new_distance, new_idx, next(tiebreaker), value))

                new_centroids = np.array([self.new_centroid(np.array(c)) for c in clusters])
                if np.linalg.norm(new_centroids - centroids) ** 2 > diff:
                    centroids = new_centroids
                else:
                    break

            labels[init_idx] = self.get_labels(points, clusters)
            if self.graph_data is not None:
                results[init_idx] = objective_function(self.graph_data, labels[init_idx])

        if self.graph_data is not None:
            best_idx = np.argmin(results)
            return labels[best_idx]
        else:
            return labels[0]


# https://arxiv.org/pdf/cond-mat/0408187.pdf
class FastModularity(object):
    def __init__(self, n_clusters, adjacency_matrix):
        self.n_clusters = n_clusters
        self.adjacency = adjacency_matrix

        N = adjacency_matrix.shape[0]

        initial_communities = [[i] for i in range(N)]
        self.communities = initial_communities # Initially, each vertex is its own community

        self.m = 0.5*np.sum(self.adjacency)
        self.degrees = self.adjacency.sum(axis=1)
        self.a = self.degrees / (2*self.m)
        self.a = np.squeeze(np.asarray(self.a))
        self.delta_Q = self.initialize_delta_Q()

        print("Initialization complete")



    def initialize_delta_Q(self):
        N = self.adjacency.shape[0]
        delta_Q = scipy.sparse.lil_matrix((N,N))
        connected_entries = np.nonzero(self.adjacency)

        print("Found {} edges".format(len(connected_entries[0])))

        for idx in range(len(connected_entries[0])):
            i = connected_entries[0][idx]
            j = connected_entries[1][idx]

            if i == j:
                continue

            val = 0.5*self.m - self.degrees[i]*self.degrees[j]/(2*self.m)**2 # Eq. 8 in the paper
            delta_Q[i,j] = delta_Q[j,i] = val

        return delta_Q

    def update_delta_Q(self, i, j):
        connections_i = np.squeeze(np.asarray((self.adjacency[i,:] == True).todense()))
        connections_j = np.squeeze(np.asarray((self.adjacency[j,:] == True).todense()))

        # Replace with more efficient implementation
        for k in range(self.adjacency.shape[0]):
            if k == i or k == j:
                continue

            # Following from Eq. 10abc from the paper
            if connections_i[k] and connections_i[k]:
                self.delta_Q[j,k] = self.delta_Q[k,j] = self.delta_Q[i,k] + self.delta_Q[j,k]
            elif connections_i[k]:
                self.delta_Q[j,k] = self.delta_Q[k,j] = self.delta_Q[i,k] - 2*self.a[j]*self.a[k]
            elif connections_j[k]:
                self.delta_Q[j,k] = self.delta_Q[k,j] = self.delta_Q[j,k] - 2*self.a[i]*self.a[k]


    def fit_predict(self, data):
        num_clusters = len(self.communities)
        while num_clusters > self.n_clusters:
            crc_Q = self.delta_Q.tocsr()
            largest_index = crc_Q.argmax()

            # i will be merged to j
            i = (int)(largest_index / self.adjacency.shape[1])
            j = largest_index % self.adjacency.shape[1]

            # Update Q
            self.update_delta_Q(i, j)
            # Delete i'th row and col from Q
            to_keep = [idx for idx in range(self.delta_Q.shape[0]) if idx != i]
            self.delta_Q = scipy.sparse.lil_matrix(scipy.sparse.csc_matrix(self.delta_Q)[:,to_keep])
            self.delta_Q = scipy.sparse.lil_matrix(scipy.sparse.csr_matrix(self.delta_Q)[to_keep,:])
            # Update a
            self.a[j] = self.a[j] + self.a[i]
            self.a = np.delete(self.a, i) # Delete i'th element from a

            # Update adjacency matrix to reflect the new community structure
            j_adjacency = self.adjacency[j,:].astype('bool')
            i_adjacency = self.adjacency[i,:].astype('bool')
            total_adjacency = i_adjacency + j_adjacency
            self.adjacency[j,:] = total_adjacency

            # Delete i'th row and col from adjacency
            self.adjacency = scipy.sparse.lil_matrix(scipy.sparse.csc_matrix(self.adjacency)[:,to_keep])
            self.adjacency = scipy.sparse.lil_matrix(scipy.sparse.csr_matrix(self.adjacency)[to_keep,:])

            # Merge communities
            self.communities[j] = list(set(self.communities[i]).union(set(self.communities[j])))
            self.communities = np.delete(self.communities, i)

            num_clusters = len(self.communities)

            if num_clusters % 10 == 0:
                print("Process: {}".format(num_clusters))

        return self.communities

        
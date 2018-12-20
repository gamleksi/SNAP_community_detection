import sys
import numpy
import re
import networkx as nx
import pandas as pd


graphid_list = ['ca-GrQc', 'Oregon-1', 'soc-Epinions1', 'web-NotreDame',
                'roadNet-CA', 'ca-HepTh', 'ca-AstroPh', 'ca-HepPh',
                'ca-CondMat']


def _read_pline(line):
    regex = re.compile(r'p (.*) (.*) (.*) (.*)', re.M | re.I)
    tokens = regex.search(line)
    return map(lambda string: string.strip(),
               map(tokens.group, range(1, 5)))


def is_graphid_valid(input_gid):
    valid = False
    for gid in graphid_list:
        if gid == input_gid:
            valid = True
            break
    return valid


def parse_clustering_file(n_g, m_g, k_g, filepath):
    error = False
    errstr = ''

    f = open(filepath, 'r')
    pline = f.readline()
    params = pline.strip(' \n').split(' ')
    if(len(params) < 5):
        error = True
        errstr = 'invalid parameter line'
        return error, errstr, -1, -1, -1, [-1], [-1]

    gid = params[1]
    n = int(params[2])
    m = int(params[3])
    k = int(params[4])

    cluster_id = numpy.zeros(n).astype(int)
    cluster_count = numpy.zeros(k).astype(int)
    for i in range(0, n):
        cluster_id[i] = -1

    if(n != n_g or m != m_g or k != k_g):
        error = True
        errstr = 'invalid parameter line'
        return error, errstr, n, m, k, cluster_id, cluster_count

    if not is_graphid_valid(gid):
        error = True
        errstr = 'invalid graphid %s' % (gid)
        return error, errstr, n, m, k, cluster_id, cluster_count

    for line in f:
        try:
            clist = [int(x) for x in line.strip('\n').split()]
        except ValueError:
            print(line)
            error = True
            errstr = 'invalid line %s' % line
            break
        
        if(len(clist) < 2):
            error = True
            errstr = 'invalid line %s' % (line)
            break
        u = clist[0]
        c = clist[1]
        if(u > n-1 or u < 0 or c < 0 or c > k-1):
            error = True
            errstr = 'invalid line %d %d' % (u, c)
            break
        if(cluster_id[u] == -1):
            cluster_id[u] = c
        else:
            error = True
            errstr = 'invalid clustering output'
            break
    f.close()
    if error:
        return error, errstr, n, m, k, cluster_id, cluster_count

    for u in range(0, n):
        cid_u = cluster_id[u]
        cluster_count[cid_u] = cluster_count[cid_u] + 1
        if(cid_u < 0):
            error = True
            errstr = 'invalid clustering output'
            break

    for q in range(0, k):
        if cluster_count[q] < 1:
            error = True
            errstr = 'No vertex in cluster %d' % q

    return error, errstr, n, m, k, cluster_id, cluster_count


def parse_graphfile(graph_path):
    sys.stdout.write("reading graphfile %s\n" % graph_path)
    sys.stdout.flush()

    with open(graph_path, 'r') as f:
        pline = f.readline()
        params = pline.strip(' \n').split(' ')
        gid = params[1]
        n = int(params[2])
        m = int(params[3])
        k = int(params[4])

    df = pd.read_csv(
        graph_path,
        sep=' ',
        names=['s', 't'],
        skiprows=1,
        header=None
    )

    edges = []
    for i, r in df.iterrows():
        edges.append((r['s'], r['t']))

    G = nx.Graph()
    G.add_edges_from(edges)
    
    return gid, n, m, k, G


def compute_cost(G, cluster_id, cluster_count):
    cost = 0
    for e in G.edges():
        u = e[0]
        v = e[1]
        if cluster_id[u] != cluster_id[v]:
            cost = cost + 1

    norm_cost = cost/min(cluster_count[0:])
    return norm_cost


def main():
    try:
        graph_path, clustering_path = sys.argv[1:]
    except ValueError:
        print("""
Invalid number of arguments
Expected command-line format: python evaluate.py {graph_path} {clustering_path}
        """)
        sys.exit(-1)
    # load graph
    gid, n, m, k, G = parse_graphfile(graph_path)

    (error, errstr, n_o, m_o, k_o,
     cluster_id, cluster_count) = parse_clustering_file(
         n, m, k, clustering_path
     )

    if error:
        print("Error encountered:", errstr)
        sys.exit(-1)
        
    cost = compute_cost(G, cluster_id, cluster_count)
    print("objective = {}".format(cost))

        
if __name__ == "__main__":
    main()

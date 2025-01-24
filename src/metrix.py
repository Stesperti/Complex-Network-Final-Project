import numpy as np
from sklearn.metrics import adjusted_rand_score
import hypergraphx as hx
import networkx as nx
import matplotlib.pyplot as plt


def compute_metrics(detected_communities, true_communities, graph=None, graph_list=None):
    """
    Compute the Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) for community detection.

    Parameters:
        detected_communities (list of lists): Detected communities.
        true_communities (list of lists): Ground truth communities.

    Returns:
        float: Adjusted Rand Index (ARI) score.
        float: Normalized Mutual Information (NMI) score.
    """
    ari = compute_ari_from_communities(detected_communities, true_communities)
    nmi = norm_mutual_information(detected_communities, true_communities)
    print(f"Adjusted Rand Index (ARI): {ari}")
    print(f"Normalized Mutual Information (NMI): {nmi}")

    if graph is not None:
        conductance_score = conductance(
            graph, convert_to_sets(detected_communities))
        print(f"Conductance: {conductance_score}")
        if graph_list is not None:
            generalized_conductance_score = generalized_conductance(
                graph_list, detected_communities)
            print(f"Generalized Conductance: {generalized_conductance_score}")
            return ari, nmi, conductance_score, generalized_conductance_score
        else:
            return ari, nmi, conductance_score

    if graph_list is not None:
        generalized_conductance_score = generalized_conductance(
            graph_list, detected_communities)
        print(f"Generalized Conductance: {generalized_conductance_score}")
        return ari, nmi, generalized_conductance_score
    return ari, nmi


def norm_mutual_information(clusters1_list, clusters2_list):
    """
    Returns the normalized mutial information between two
    potentially overlapping clustering.

    The mutual information is normalized by the max of the 
    two individual entropies.

    .. math::
        NMI = 0.5*(H(C1)-H(C1|C2)+H(C2)-H(C2|C1))/max(H(C1),H(C2))

    inputs must be lists of node sets

    See :
    [1] Lancichinetti, A., Fortunato, S., & Kertész, J. (2009). 
        Detecting the overlapping and hierarchical community structure in 
        complex networks. New journal of physics, 11(3), 033015.    
    [2] McDaid, A. F., Greene, D., & Hurley, N. (2011). 
        Normalized mutual information to evaluate overlapping community finding 
        algorithms. arXiv preprint arXiv:1110.2515.    
    """

    clusters1 = convert_to_sets(clusters1_list)
    clusters2 = convert_to_sets(clusters2_list)

    # print(clusters1)
    # print(clusters2)

    # num nodes
    N = len(set.union(*clusters1))
    # Entropy information
    def h(p): return -1*p*np.log2(p) if p > 0 else 0

    def cond_entropy(clusters1, clusters2):
        Hi2 = []
        for i, clust1 in enumerate(clusters1):
            Hij = []
            for j, clust2 in enumerate(clusters2):
                intersect = len(clust1.intersection(clust2))
                union = len(clust1.union(clust2))
                p11 = intersect/N  # prob to be in 1 & 2
                # prob to be in 1 but not in 2
                p10 = (len(clust1) - intersect)/N
                # prob to be in 2 but not in 1
                p01 = (len(clust2) - intersect)/N
                p00 = (N - union)/N  # prob to not be in 1 nor 2
                p2 = len(clust2)/N
                # entropies
                h11 = h(p11)
                h00 = h(p00)
                h01 = h(p01)
                h10 = h(p10)
                # conditional entropy H(i|j)
                if h11 + h00 > h01 + h10:  # refering to equation B.14
                    Hij.append(h11 + h00 + h01 + h10 - h(p2) - h(1-p2))

            if len(Hij) > 0:  # B.9 equation
                # cond entropy H(i|2)
                Hi2.append(min(Hij))
            else:
                p1 = len(clust1)/N
                Hi2.append(h(p1) + h(1-p1))

        return sum(Hi2)

    def entropy(clusters):
        return sum(h(len(clust)/N)+h((1-len(clust)/N)) for clust in clusters)

    # Mutual information
    H1 = entropy(clusters1)
    H2 = entropy(clusters2)
    # print("entropy1", H1)
    # print("entropy2", H2)

    # print(cond_entropy(clusters1, clusters2), cond_entropy(clusters2, clusters1))

    MI = 0.5*(H1 - cond_entropy(clusters1, clusters2) +
              H2 - cond_entropy(clusters2, clusters1))

    return MI/max((H1, H2))


def convert_to_sets(cluster_list):
    """
    Converts a list of lists into a list of sets.
    """
    return [set(cluster) for cluster in cluster_list]


def conductance(simple_graph, partition):

    sum_conductance = 0
    for i in range(len(partition)):
        sum_conductance += nx.algorithms.cuts.conductance(
            simple_graph, partition[i])

    return sum_conductance/len(partition)


def generalized_conductance(hypergraph, partition):
    """
    Compute the generalized conductance of a partition of a hypergraph.

    - Hypegraph: list of lists of nodes
    - Partition: list of lists of nodes
    """
    maximum_edge = max([len(x) for x in hypergraph])
    conductance = 0
    for i in range(2, maximum_edge+1):
        conductance_i = 0
        hypergraph_i = [x for x in hypergraph if len(x) == i]
        if len(hypergraph_i) == 0:
            continue
        for elements_partition in range(len(partition)):
            cut = 0
            vol_S = 0
            for hyperedge in hypergraph_i:

                intersection = len(set(hyperedge).intersection(
                    set(partition[elements_partition])))
                if intersection > 0 and intersection < i:
                    cut += (i - intersection)

                vol_S += intersection

            if vol_S == 0 and cut == 0:
                conductance_i += 0
            elif vol_S == 0 and cut != 0:
                print("cut", cut)
                print("vol_S", vol_S)
                print("ERROR")
            else:
                conductance_i += cut/vol_S

        conductance += conductance_i
    return conductance/len(partition)


def compute_ari_from_communities(detected_communities, true_communities):
    """
    Compute the Adjusted Rand Index (ARI) for community detection.

    Parameters:
        detected_communities (list of lists): Detected communities.
        true_communities (list of lists): Ground truth communities.

    Returns:
        float: Adjusted Rand Index (ARI) score.
    """
    # Create node-to-community label mappings
    detected_labels = {}
    true_labels = {}

    for idx, community in enumerate(detected_communities):
        for node in community:
            detected_labels[node] = idx

    for idx, community in enumerate(true_communities):
        for node in community:
            true_labels[node] = idx

    # Ensure both detected and true labels cover the same set of nodes
    all_nodes = set(detected_labels.keys()).union(set(true_labels.keys()))
    labels_true = [true_labels.get(node, -1) for node in all_nodes]
    labels_pred = [detected_labels.get(node, -1) for node in all_nodes]

    # Compute ARI
    ari_score = adjusted_rand_score(labels_true, labels_pred)
    return ari_score


def markov_stability_hyperedges(hypergraph, partition_matrix, n_nodes, figure = True):
    """
    Compute the Markov Stability of a hypergraph

    Parameters
    ----------
    hypergraph : List of lists of integers
        The hypergraph to be analyzed
    partition : List of integers
    """

    e_matrix = np.zeros((n_nodes, len(hypergraph)))

    for i, hyperedge in enumerate(hypergraph):
        for node in hyperedge:
            e_matrix[node, i] = 1

    A = e_matrix @ e_matrix.T
    C_hat = e_matrix.T @ e_matrix
    np.fill_diagonal(C_hat, 0)

    eCe = e_matrix @ C_hat @ e_matrix.T
    k_H = eCe - A
    T = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        denominator = np.sum(k_H[i, :])
        for j in range(n_nodes):
            T[i, j] = k_H[i, j] / denominator

    stationary_denominator = np.sum(k_H)

    stationary = np.zeros(n_nodes)
    for i in range(n_nodes):
        stationary[i] = np.sum(k_H[i, :]) / stationary_denominator

    stationary_matrix = np.diag(stationary)
    
    results = []   
    for t in range(1, 10):
        T_t = np.linalg.matrix_power(T, t)
        markov_stability = partition_matrix.T @ (
                    stationary_matrix @ T_t - stationary.T @ stationary) @ partition_matrix
        output = np.trace(markov_stability)
        results.append(output)

    if figure:
        plt.figure()
        plt.title('Markov Stability')
        plt.plot(results)
        plt.show()

    return results


# Example usage
if __name__ == "__main__":
    Hypegraph = [[1, 2, 4, 5], [3, 4], [2, 5]]
    detected = [[1, 2, 4, 5], [3]]
    true = [[1, 2, 4, 5], [3]]

    ari = compute_ari_from_communities(detected, true)
    print(f"Adjusted Rand Index (ARI): {ari}")

    gen_cond = generalized_conductance(Hypegraph, detected)
    print(f"Generalized Conductance: {gen_cond}")

import networkx as nx
import scipy.special

def k_clique_reduction(hypergraph, weight=False):
    """
    Replace the hyperedges of size k with a k-clique.
    - hypergraph: list of lists, where each sublist is a hyperedge.
    - weight: boolean, if True, edge weights are calculated based on clique size.
    """
    K_clique_graph = nx.Graph()
    for i in hypergraph:
        size = len(i)
        if size == 2:
            if K_clique_graph.has_edge(i[0], i[1]):
                K_clique_graph[i[0]][i[1]]['weight'] += 1
            else:
                K_clique_graph.add_edge(i[0], i[1], weight=1)
        else:
            for j in range(size):
                for k in range(j + 1, size):
                    if weight:
                        weight_edge = float(1/ scipy.special.binom(size, 2))  
                    else:
                        weight_edge = 1
                    
                    if K_clique_graph.has_edge(i[j], i[k]):
                        K_clique_graph[i[j]][i[k]]['weight'] += weight_edge
                    else:
                        K_clique_graph.add_edge(i[j], i[k], weight=weight_edge)
    
    return K_clique_graph


    
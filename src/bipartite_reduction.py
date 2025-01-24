import networkx as nx
import scipy.special


def bipartite_reduction(hypergraph, weight = False):
    """
    Converts a hypergraph into its bipartite representation.

    Parameters:
        hypergraph (list of lists): A hypergraph represented as a list of edges, 
                                    where each edge is a list of nodes.

    Returns:
        nx.Graph: A bipartite graph representation of the hypergraph.
    """
    Bipartite_graph = nx.Graph()

    for i, edge in enumerate(hypergraph):
        node_edge = 'e' + str(i)
        size = len(edge)
        if weight:
            weight_edge = float(1/ scipy.special.binom(size, 2))
        else:
            weight_edge = 1
        for node in edge:
            Bipartite_graph.add_edge(node_edge, node, weight = weight_edge)

    return Bipartite_graph

if __name__ == "__main__":
    hypergraph = [[1, 2, 3], [2, 3, 4], [1, 4]]

    Bipartite_graph = bipartite_reduction(hypergraph)

    print("Edges of the bipartite graph:")
    print(Bipartite_graph.edges())
    print("Nodes of the bipartite graph:")
    print(Bipartite_graph.nodes())
        




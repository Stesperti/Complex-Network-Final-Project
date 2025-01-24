import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import community 

import random
import numpy as np
   
def modularity_communities_k_clique(G, weight=None):
    
    if weight is None:
        #print("Using unweighted modularity")
        #edges = G.edges(data=True)
        #print({(x[0], x[1]): x[2]['weight'] for x in edges if 'weight' in x[2]})
        partition = community.best_partition(G, weight='weight')
    else:
        #print("Using weighted modularity")
        partition = community.best_partition(G, weight='weight')
        #edges = G.edges(data=True)
        #print({(x[0], x[1]): x[2]['weight'] for x in edges if 'weight' in x[2]})
    
    grouped = {}
    for key, value in partition.items():
        grouped.setdefault(value, []).append(key)
    
    if 0 not in G.nodes():
        print("problem in the data")

    if len(G.nodes()) != max(G.nodes())+1:
        print("problem in the data maximum")
  
        print("problem in the data maximum")
        print(len(G.nodes()))
        print(max(G.nodes()))
    # Initialize modularity matrix
    matrix_modularity = np.zeros((len(G.nodes()), len(grouped)))
    
    for node, community_id in partition.items():
        
        matrix_modularity[node][community_id] = 1

    return list(grouped.values()), matrix_modularity
   
def modularity_communities_bipartite(G, weight=None):
    
    if weight is None:
        #print("Using unweighted modularity")
        #edges = G.edges(data=True)
        #print({(x[0], x[1]): x[2]['weight'] for x in edges if 'weight' in x[2]})
        partition = community.best_partition(G, weight='weight')
    else:
        #print("Using weighted modularity")
        partition = community.best_partition(G, weight='weight')
        #edges = G.edges(data=True)
        #print({(x[0], x[1]): x[2]['weight'] for x in edges if 'weight' in x[2]})
    
    grouped = {}
    for key, value in partition.items():
        grouped.setdefault(value, []).append(key)
    

    return list(grouped.values()) 


def plot_communities_given_partition(graph, partition):
    plt.figure(figsize=(10, 10))
    
    # Compute positions for nodes
    pos = nx.spring_layout(graph)
    
    # Create a mapping from node to community index
    node_to_community = {}
    for community_index, community in enumerate(partition):
        for node in community:
            node_to_community[node] = community_index

    # Assign colors based on community indices
    cmap = plt.get_cmap('viridis')
    #print(partition)
    #print(node_to_community)
    #print(graph.nodes())
    #if 0 in graph.nodes():
    #    print("0 in nodes")
    colors = [node_to_community[node] for node in graph.nodes()]

    # Draw nodes and edges
    nx.draw_networkx_nodes(graph, pos, node_color=colors, cmap=cmap, node_size=100)
    nx.draw_networkx_edges(graph, pos, alpha=0.5)

    # Add labels to nodes
    labels = {node: str(node) for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels, font_size=8)

    plt.show()


def modularity_community_list(graph, partition = None):
    if partition is None:
        partition = modularity_communities_k_clique(graph)

    unique_values = set(partition.values())
    communities = [
        [key for key, val in partition.items() if val == value]
        for value in unique_values
    ]
    communities = [
        sorted(int(num) for num in community)
        for community in communities
    ]
    return communities


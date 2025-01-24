import numpy as np
import pandas as pd

def import_txt_comma_hypergraph(file_path):
    hypergraph = []

    with open(file_path, 'r') as f:
        for line in f:
            # Split by commas, then convert each part to integers
            edges = list(map(int, line.strip().split(',')))
            edges_from_0 = [x-1 for x in edges]
            hypergraph.append(edges_from_0)
    return hypergraph


def import_txt_true_one_community(file_path):
    true_community = {}
    true_matrix = {}
    

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if int(line.strip()) not in true_community:
                true_community[int(line.strip())] = []
            true_community[int(line.strip())].append(i)
            true_matrix[i] = int(line.strip())

        number_community = len(true_community)
        matrix = np.zeros((len(true_matrix), number_community))
        for i in range(len(true_matrix)):
            matrix[i][true_matrix[i]-1] = 1
    
    return true_community, matrix


def import_from_npz(file_path):
    data = np.load(file_path, allow_pickle=True)
    hyperedges = data['hyperedges']
    list_hyperedges = []    
    for i in range(len(hyperedges)):
        list_hyperedges.append([int(x) for x in hyperedges[i]])

    return list_hyperedges




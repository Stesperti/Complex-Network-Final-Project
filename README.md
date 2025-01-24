# **Final Project: Complex Networks Theory and Applications (MAT933)**

Istitution: University of Zurich and Università degli studi di Torino  
Autor: **Stefano Sperti**

## Title : Measuring the Effictiveness of Hypergraph Community Detection

Understanding community structures is crucial for uncovering patterns in complex networks. While traditional clustering methods focus on pairwise relationships, higher-order graphs capture richer structural information, which can be used to improve community detection strategies. This project investigates the performance of community detection algorithms in the HypergraphX package, evaluating them with various quality measures and comparing them with community detection algorithms applied to the graph reduction of the hypergraph. Our results show that these algorithms provide limited improvements when the number of partitions is the same as in the graph. However, these algorithms are flexible, as they take the number of communities to cluster as an input and perform well when the true number of communities is given, shifting the problem to a model selection challenge. For this reason, we implement different types of measures for evaluating the quality of clusters with and without ground-truth communities. To achieve this, we extend conductance to a hyperedge conductance setting, identify its limitations, and propose a generalized Markov stability measure for hypergraphs. We compare various measures of community detection effectiveness and analyze their characteristics.

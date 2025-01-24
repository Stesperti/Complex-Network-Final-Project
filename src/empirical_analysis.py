import hypergraphx as hx
from hypergraphx.core.hypergraph import Hypergraph
from hypergraphx.utils import normalize_array, calculate_permutation_matrix
from hypergraphx.communities.hy_sc.model import HySC
from hypergraphx.communities.hy_mmsbm.model import HyMMSBM
from hypergraphx.viz import draw_communities


import networkx as nx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from k_clique_reduction import k_clique_reduction
from metrix import *
from modularity_graph import *
from bipartite_reduction import *
from methods_clustering import *
from evaluation_clustering import *


def empirical_analysis_project(hypergraph, true_partition_hypergraph, true_matrix_hypergraph, min_number, max_number, K, true_number_community, seed=22, n_realizations=10, assortative=True, max_iter=500 ):

    #############################
    # K -clique reduction
    #############################

    true_partition_hypergraph_list = list(true_partition_hypergraph.values())
    n_nodes = max([i for hyperedge in hypergraph for i in hyperedge]) + 1

    distribution_length = [len(edge) for edge in hypergraph]
    plt.figure()
    counts, bins, patches = plt.hist(distribution_length, bins=range(2, max(distribution_length) + 1), alpha=0.75)

    # Adding annotations on top of each bar
    for count, patch in zip(counts, patches):
        height = patch.get_height()
        plt.text(patch.get_x() + patch.get_width() / 2, height, str(int(count)), ha='center', va='bottom')

    plt.title('Distribution of the size of the hyperedges')
    plt.xlabel('Size of the hyperedges')
    plt.ylabel('Number of hyperedges')
    plt.show()
        

    if n_nodes > 500:
        n_realizations = 5
    Hypergraph_package = hx.Hypergraph(hypergraph)


    cmap = sns.color_palette("husl", n_colors=true_number_community)
    col = {k: matplotlib.colors.to_hex(
        cmap[k], keep_alpha=False) for k in np.arange(true_number_community)}
    plt.figure(facecolor='white')
    draw_communities(hypergraph=Hypergraph_package, u=true_matrix_hypergraph, col=col, with_node_labels=False, 
                     title='True partition', node_size = 0.06)
    
    plt.show()



    print("-----------------------------------------------------------------------------------")
    print("K -clique reduction")
    print("-----------------------------------------------------------------------------------")
    k_reduction_hypergraph = k_clique_reduction(hypergraph)

    if len(k_reduction_hypergraph.edges) < 500:
        nx.draw(k_reduction_hypergraph, with_labels=False)
        plt.show()

    partition_k_reduction_hypergraph, k_reduction_matrix_primary = modularity_communities_k_clique(
        k_reduction_hypergraph)

    plot_communities_given_partition(
        k_reduction_hypergraph, partition_k_reduction_hypergraph)

    compute_metrics(partition_k_reduction_hypergraph,
                    true_partition_hypergraph_list, k_reduction_hypergraph, hypergraph)

    markov_stability_hyperedges(
        hypergraph, k_reduction_matrix_primary, n_nodes)
    print("Number of partitions of K -clique reduction: ",
          len(partition_k_reduction_hypergraph))

    #############################
    # K -clique reduction WEIGHT
    #############################

    print("-----------------------------------------------------------------------------------")
    print("K -clique reduction weighting the hyperedges")
    print("-----------------------------------------------------------------------------------")

    k_reduction_hypergraph = k_clique_reduction(hypergraph, weight=True)

    partition_k_reduction_hypergraph_weight, k_reduction_matrix_primary_weight = modularity_communities_k_clique(
        k_reduction_hypergraph, weight=True)

    true_partition_hypergraph_list = list(true_partition_hypergraph.values())

    compute_metrics(partition_k_reduction_hypergraph_weight,
                    true_partition_hypergraph_list, k_reduction_hypergraph, hypergraph)

    markov_stability_hyperedges(hypergraph, k_reduction_matrix_primary_weight, n_nodes)

    print("Number of partitions of K -clique reduction weighting the hyperedges: ",
          len(partition_k_reduction_hypergraph_weight))

    #############################
    # Bipartite representation
    #############################

    print("-----------------------------------------------------------------------------------")
    print("Bipartite representation")
    print("-----------------------------------------------------------------------------------")

    bipartite_hypergraph = bipartite_reduction(hypergraph)

    partition_bipartite_hypergraph = modularity_communities_bipartite(
        bipartite_hypergraph)

    #print(partition_bipartite_hypergraph)
    filtered_partition_bipartite_hypergraph = [[elem for elem in sublist if isinstance(
        elem, int)] for sublist in partition_bipartite_hypergraph]
    #print(filtered_partition_bipartite_hypergraph)

    compute_metrics(filtered_partition_bipartite_hypergraph,
                    true_partition_hypergraph_list, k_reduction_hypergraph, hypergraph)


    print("Number of partitions: ", len(filtered_partition_bipartite_hypergraph))

    #############################
    # Weighted Bipartite representation
    #############################

    print("-----------------------------------------------------------------------------------")
    print("Weighted Bipartite representation")
    print("-----------------------------------------------------------------------------------")

    bipartite_hypergraph = bipartite_reduction(hypergraph, weight=True)

    partition_bipartite_hypergraph = modularity_communities_bipartite(
        bipartite_hypergraph, weight=True)

    filtered_partition_bipartite_hypergraph = [[elem for elem in sublist if isinstance(
        elem, int)] for sublist in partition_bipartite_hypergraph]

    compute_metrics(filtered_partition_bipartite_hypergraph,
                    true_partition_hypergraph_list, k_reduction_hypergraph, hypergraph)

    print("Number of partitions: ", len(filtered_partition_bipartite_hypergraph))

    #############################
    # Spectral clustering K = modularity
    #############################

    print("-----------------------------------------------------------------------------------")
    print("Spectral clustering with number of community find with modularity")
    print("-----------------------------------------------------------------------------------")

    model = HySC(
        seed=seed,
        n_realizations=n_realizations
    )


    u_HySC = model.fit(
        Hypergraph_package,
        K=len(partition_k_reduction_hypergraph),
        weighted_L=True
    )

    spectral_partition_hypergraph = matrix_to_community_list(u_HySC)


    compute_metrics(spectral_partition_hypergraph,
                    true_partition_hypergraph_list, k_reduction_hypergraph, hypergraph)

    markov_stability_hyperedges(hypergraph, u_HySC, n_nodes)

    #############################
    # HyMMSBM
    #############################

    print("-----------------------------------------------------------------------------------")
    print("HyMMSBM with number of community of modularity")
    print("-----------------------------------------------------------------------------------")

    np.random.seed(seed)
    random.seed(seed)

    # Train some models with different random initializations, choose the best one in terms of likelihood
    best_model = None
    best_loglik = float("-inf")
    for j in range(n_realizations):
        model = HyMMSBM(
            K=len(partition_k_reduction_hypergraph),
            assortative=assortative
        )
        model.fit(
            Hypergraph_package,
            n_iter=max_iter
        )

        log_lik = model.log_likelihood(Hypergraph_package)
        if log_lik > best_loglik:
            best_model = model
            best_loglik = log_lik

    u_HyMMSBM = best_model.u
    w_HyMMSBM = best_model.w

    HyMMSBM_community_given_hypergraph, HyMMSBM_community_given_hypergraph_matrix = select_maximum_row_community(
        u_HyMMSBM)


    compute_metrics(HyMMSBM_community_given_hypergraph,
                    true_partition_hypergraph_list, k_reduction_hypergraph, hypergraph)

    markov_stability_hyperedges(
        hypergraph, HyMMSBM_community_given_hypergraph_matrix, n_nodes)

    #############################
    # Plotting
    #############################
    u_HySC_modularity = u_HySC.copy()
    HyMMSBM_community_given_hypergraph_matrix_modularity = HyMMSBM_community_given_hypergraph_matrix.copy()
    print("-----------------------------------------------------------------------------------")
    print("Comparison of the different methods")
    print("-----------------------------------------------------------------------------------")

    Color_max = max(K, len(true_partition_hypergraph_list),
                    k_reduction_matrix_primary.shape[1])

    cmap = sns.color_palette("tab20", n_colors=Color_max, desat=0.7)
    col = {k: matplotlib.colors.to_hex(
        cmap[k], keep_alpha=False) for k in np.arange(Color_max)}

    plt.figure(figsize=(14, 14))
    plt.subplot(2, 2, 1)
    ax = plt.gca()
    draw_communities(hypergraph=Hypergraph_package, u=true_matrix_hypergraph, col=col,
                     ax=ax, with_node_labels=False, title='True partition', node_size = 0.06)
    plt.subplot(2, 2, 2)
    ax = plt.gca()
    draw_communities(hypergraph=Hypergraph_package, u=k_reduction_matrix_primary, col=col,
                     ax=ax, with_node_labels=False, title='K-expantion represenation', node_size = 0.06)
    plt.subplot(2, 2, 3)
    ax = plt.gca()
    draw_communities(hypergraph=Hypergraph_package, u=u_HySC, col=col, ax=ax,
                     with_node_labels=False, title='Hypergraph Spectral Clustering', node_size = 0.06)
    plt.subplot(2, 2, 4)
    ax = plt.gca()
    draw_communities(hypergraph=Hypergraph_package, u=HyMMSBM_community_given_hypergraph_matrix,
                     col=col, ax=ax, with_node_labels=False, title='Hy-MMSBM', node_size = 0.06)

    plt.show()


    #############################
    # Spectral clustering
    #############################

    print("-----------------------------------------------------------------------------------")
    print("Spectral clustering with true number of clusters")
    print("-----------------------------------------------------------------------------------")


    model = HySC(
        seed=seed,
        n_realizations=n_realizations
    )

    u_HySC = model.fit(
        Hypergraph_package,
        K=K,
        weighted_L=True
    )

    spectral_partition_hypergraph = matrix_to_community_list(u_HySC)

    P = calculate_permutation_matrix(
        u_ref=true_matrix_hypergraph, u_pred=u_HySC)
    u_HySC = np.dot(u_HySC, P)

    compute_metrics(spectral_partition_hypergraph,
                    true_partition_hypergraph_list, k_reduction_hypergraph, hypergraph)

    markov_stability_hyperedges(hypergraph, u_HySC, n_nodes)

    #############################
    # HyMMSBM
    #############################

    print("-----------------------------------------------------------------------------------")
    print("HyMMSBM with true number of clusters")
    print("-----------------------------------------------------------------------------------")

    np.random.seed(seed)
    random.seed(seed)

    # Train some models with different random initializations, choose the best one in terms of likelihood
    best_model = None
    best_loglik = float("-inf")
    for j in range(n_realizations):
        model = HyMMSBM(
            K=K,
            assortative=assortative
        )
        model.fit(
            Hypergraph_package,
            n_iter=max_iter
        )

        log_lik = model.log_likelihood(Hypergraph_package)
        if log_lik > best_loglik:
            best_model = model
            best_loglik = log_lik

    u_HyMMSBM = best_model.u
    w_HyMMSBM = best_model.w

    HyMMSBM_community_given_hypergraph, HyMMSBM_community_given_hypergraph_matrix = select_maximum_row_community(
        u_HyMMSBM)
    
    P = calculate_permutation_matrix(
        u_ref=true_matrix_hypergraph, u_pred=HyMMSBM_community_given_hypergraph_matrix)
    HyMMSBM_community_given_hypergraph_matrix = np.dot(
        HyMMSBM_community_given_hypergraph_matrix, P)

    compute_metrics(HyMMSBM_community_given_hypergraph,
                    true_partition_hypergraph_list, k_reduction_hypergraph, hypergraph)
            
    markov_stability_hyperedges(hypergraph, HyMMSBM_community_given_hypergraph_matrix, n_nodes)

    #############################
    # Plotting
    #############################

    print("-----------------------------------------------------------------------------------")
    print("Comparison of the different methods")
    print("-----------------------------------------------------------------------------------")

    Color_max = max(K, len(true_partition_hypergraph_list),
                    k_reduction_matrix_primary.shape[1])

    cmap = sns.color_palette("tab20", n_colors=Color_max, desat=0.7)
    col = {k: matplotlib.colors.to_hex(
        cmap[k], keep_alpha=False) for k in np.arange(Color_max)}

    plt.figure(figsize=(14, 14))
    plt.subplot(2, 2, 1)
    ax = plt.gca()
    draw_communities(hypergraph=Hypergraph_package, u=true_matrix_hypergraph, col=col,
                     ax=ax, with_node_labels=False, title='True partition', node_size = 0.06)
    plt.subplot(2, 2, 2)
    ax = plt.gca()
    draw_communities(hypergraph=Hypergraph_package, u=k_reduction_matrix_primary, col=col,
                     ax=ax, with_node_labels=False, title='k-expantion represenation', node_size = 0.06)
    plt.subplot(2, 2, 3)
    ax = plt.gca()
    draw_communities(hypergraph=Hypergraph_package, u=u_HySC, col=col, ax=ax,
                     with_node_labels=False, title='Hypergraph Spectral Clustering', node_size = 0.06)
    plt.subplot(2, 2, 4)
    ax = plt.gca()
    draw_communities(hypergraph=Hypergraph_package, u=HyMMSBM_community_given_hypergraph_matrix,
                     col=col, ax=ax, with_node_labels=False, title='Hy-MMSBM', node_size = 0.06)

    plt.show()



        #############################
    # NEW PLOTTING
    #############################

    print("-----------------------------------------------------------------------------------")
    print("Comparison of the different methods")
    print("-----------------------------------------------------------------------------------")

    Color_max = max(K, len(true_partition_hypergraph_list),
                    k_reduction_matrix_primary.shape[1])

    modularity_number_community = k_reduction_matrix_primary.shape[1]
    
    cmap = sns.color_palette("tab20", n_colors=Color_max, desat=0.7)
    col = {k: matplotlib.colors.to_hex(
        cmap[k], keep_alpha=False) for k in np.arange(Color_max)}

    plt.figure(figsize=(18, 12))  # Adjust the figure size to fit 6 plots

    # First plot
    plt.subplot(2, 3, 1)  # 2 rows, 3 columns, first plot
    ax = plt.gca()
    draw_communities(hypergraph=Hypergraph_package, u=true_matrix_hypergraph, col=col,
                    ax=ax, with_node_labels=False, title='True partition', node_size=0.06)

    # Second plot
    plt.subplot(2, 3, 2)  # 2 rows, 3 columns, second plot
    ax = plt.gca()
    draw_communities(hypergraph=Hypergraph_package, u=k_reduction_matrix_primary, col=col,
                    ax=ax, with_node_labels=False, title='k-expansion representation', node_size=0.06)

    # Third plot
    plt.subplot(2, 3, 3)  # 2 rows, 3 columns, third plot
    ax = plt.gca()
    draw_communities(hypergraph=Hypergraph_package, u=u_HySC_modularity, col=col, ax=ax,
                    with_node_labels=False, title=f'Spectral Clustering {modularity_number_community} community', node_size=0.06)

    # Fourth plot
    plt.subplot(2, 3, 4)  # 2 rows, 3 columns, fourth plot
    ax = plt.gca()
    draw_communities(hypergraph=Hypergraph_package, u=HyMMSBM_community_given_hypergraph_matrix_modularity,
                    col=col, ax=ax, with_node_labels=False, title=f'Hy-MMSBM {modularity_number_community} community', node_size=0.06)

    # Fifth plot
    plt.subplot(2, 3, 5)  # 2 rows, 3 columns, fifth plot
    ax = plt.gca()
    draw_communities(hypergraph=Hypergraph_package, u=u_HySC,
                    col=col, ax=ax, with_node_labels=False, title=f'Spectral Clustering {true_number_community} community', node_size=0.06)

    # Sixth plot
    plt.subplot(2, 3, 6)  # 2 rows, 3 columns, sixth plot
    ax = plt.gca()
    draw_communities(hypergraph=Hypergraph_package, u=HyMMSBM_community_given_hypergraph_matrix,
                    col=col, ax=ax, with_node_labels=False, title=f'Hy-MMSBM {true_number_community} community', node_size=0.06)

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

    ################
    # Plot senate 
    ##########
    if assortative == False:
        best_model = None
        best_loglik = float("-inf")
        for j in range(n_realizations):
            model = HyMMSBM(
                K=K,
                assortative=True
            )
            model.fit(
                Hypergraph_package,
                n_iter=max_iter
            )

            log_lik = model.log_likelihood(Hypergraph_package)
            if log_lik > best_loglik:
                best_model = model
                best_loglik = log_lik

        u_HyMMSBM = best_model.u
        w_HyMMSBM = best_model.w

        HyMMSBM_community_TRUE_NUMBER, HyMMSBM_community_Matrix_TRUE_NUMBER = select_maximum_row_community(
            u_HyMMSBM)
        
        best_model = None
        best_loglik = float("-inf")
        for j in range(n_realizations):
            model = HyMMSBM(
                K=len(partition_k_reduction_hypergraph),
                assortative=True
            )
            model.fit(
                Hypergraph_package,
                n_iter=max_iter
            )

            log_lik = model.log_likelihood(Hypergraph_package)
            if log_lik > best_loglik:
                best_model = model
                best_loglik = log_lik

        u_HyMMSBM = best_model.u
        w_HyMMSBM = best_model.w

        HyMMSBM_community_TRUE_NUMBER_modularity, HyMMSBM_community_Matrix_TRUE_NUMBER_modularity = select_maximum_row_community(
            u_HyMMSBM)
        print("-----------------------------------------------------------------------------------")
        print("Different assortative")
        print("-----------------------------------------------------------------------------------")

        Color_max = max(K, len(true_partition_hypergraph_list),
                        k_reduction_matrix_primary.shape[1])

        modularity_number_community = k_reduction_matrix_primary.shape[1]
        
        cmap = sns.color_palette("tab20", n_colors=Color_max, desat=0.7)
        col = {k: matplotlib.colors.to_hex(
            cmap[k], keep_alpha=False) for k in np.arange(Color_max)}

        plt.figure(figsize=(18, 12))  # Adjust the figure size to fit 6 plots

        # First plot
        plt.subplot(2, 3, 1)  # 2 rows, 3 columns, first plot
        ax = plt.gca()
        draw_communities(hypergraph=Hypergraph_package, u=true_matrix_hypergraph, col=col,
                        ax=ax, with_node_labels=False, title='True partition', node_size=0.06)

        # Second plot
        plt.subplot(2, 3, 2)  # 2 rows, 3 columns, second plot
        ax = plt.gca()
        draw_communities(hypergraph=Hypergraph_package, u=k_reduction_matrix_primary, col=col,
                        ax=ax, with_node_labels=False, title='k-expansion representation', node_size=0.06)

        # Third plot
        plt.subplot(2, 3, 3)  # 2 rows, 3 columns, third plot
        ax = plt.gca()
        draw_communities(hypergraph=Hypergraph_package, u=HyMMSBM_community_Matrix_TRUE_NUMBER_modularity, col=col, ax=ax,
                        with_node_labels=False, title=f'Hy-MMSBM Assortative {modularity_number_community} community', node_size=0.06)

        # Fourth plot
        plt.subplot(2, 3, 4)  # 2 rows, 3 columns, fourth plot
        ax = plt.gca()
        draw_communities(hypergraph=Hypergraph_package, u=HyMMSBM_community_given_hypergraph_matrix_modularity,
                        col=col, ax=ax, with_node_labels=False, title=f'Hy-MMSBM Disassortative {modularity_number_community} community', node_size=0.06)

        # Fifth plot
        plt.subplot(2, 3, 5)  # 2 rows, 3 columns, fifth plot
        ax = plt.gca()
        draw_communities(hypergraph=Hypergraph_package, u=HyMMSBM_community_Matrix_TRUE_NUMBER,
                        col=col, ax=ax, with_node_labels=False, title=f'Hy-MMSBM Assortative  {true_number_community} community', node_size=0.06)

        # Sixth plot
        plt.subplot(2, 3, 6)  # 2 rows, 3 columns, sixth plot
        ax = plt.gca()
        draw_communities(hypergraph=Hypergraph_package, u=HyMMSBM_community_given_hypergraph_matrix,
                        col=col, ax=ax, with_node_labels=False, title=f'Hy-MMSBM Disassortative  {true_number_community} community', node_size=0.06)

        # Adjust layout to avoid overlap
        plt.tight_layout()
        plt.show()
    

    #############################
    # Spectral clustering choose of K
    #############################

    print("-----------------------------------------------------------------------------------")
    print("Spectral clustering choose of K")
    print("-----------------------------------------------------------------------------------")

    model = HySC(
        seed=seed,
        n_realizations=n_realizations
    )

    results = []
    ari_scores = []
    nmi_scores = []
    conductance_scores = []
    generalized_conductance_scores = []
    generalized_markov_stability_scores = []
    for number_community in range(min_number, max_number):
        u_HySC = model.fit(
            Hypergraph_package,
            K=number_community,
            weighted_L=True
        )
        partition_this_time = matrix_to_community_list(u_HySC)
        results.append(partition_this_time)
        print("Number of communities: ", number_community)
        ari, nmi, condscore, gen_cond_score = compute_metrics(
            partition_this_time, true_partition_hypergraph_list, k_reduction_hypergraph, hypergraph)
        gen_markov = markov_stability_hyperedges(hypergraph, u_HySC, n_nodes,False)
        
        ari_scores.append(ari)
        nmi_scores.append(nmi)
        conductance_scores.append(condscore)
        generalized_conductance_scores.append(gen_cond_score)
        generalized_markov_stability_scores.append(gen_markov)

   
    plotting_measure(ari_scores, min_number, max_number, true_number_community, label='ARI index')
    plotting_measure(nmi_scores, min_number, max_number, true_number_community, label='Normalized mutual information')
    plotting_measure(conductance_scores, min_number, max_number, true_number_community, label='Conductance')
    plotting_measure(generalized_conductance_scores, min_number, max_number, true_number_community, label='Generalized Conductance')
    plotting_markov_stability(generalized_markov_stability_scores, min_number, max_number, true_number_community)

    #############################
    # HyMMSBM choose of K
    #############################

    print("-----------------------------------------------------------------------------------")
    print("HyMMSBM choose of K")
    print("-----------------------------------------------------------------------------------")

    results = []
    ari_scores = []
    nmi_scores = []
    conductance_scores = []
    generalized_conductance_scores = []
    generalized_markov_stability_scores = []
    for number_community in range(min_number, max_number):
        best_model = None
        best_loglik = float("-inf")
        for j in range(n_realizations):
            model = HyMMSBM(
                K=K,
                assortative=assortative
            )
            model.fit(
                Hypergraph_package,
                n_iter=max_iter
            )

            log_lik = model.log_likelihood(Hypergraph_package)
            if log_lik > best_loglik:
                best_model = model
                best_loglik = log_lik

        u_HyMMSBM = best_model.u
        w_HyMMSBM = best_model.w

        HyMMSBM_community_given_hypergraph, HyMMSBM_community_given_hypergraph_matrix = select_maximum_row_community(
            u_HyMMSBM)

        results.append(HyMMSBM_community_given_hypergraph)

        ari, nmi, condscore, gen_cond_score = compute_metrics(HyMMSBM_community_given_hypergraph,
                        true_partition_hypergraph_list, k_reduction_hypergraph, hypergraph)
        gen_markov = markov_stability_hyperedges(hypergraph, HyMMSBM_community_given_hypergraph_matrix, n_nodes,False)

        ari_scores.append(ari)
        nmi_scores.append(nmi)
        conductance_scores.append(condscore)
        generalized_conductance_scores.append(gen_cond_score)
        generalized_markov_stability_scores.append(gen_markov)

    
    plotting_measure(ari_scores, min_number, max_number, true_number_community, label='ARI index')
    plotting_measure(nmi_scores, min_number, max_number, true_number_community, label='Normalized mutual information')
    plotting_measure(conductance_scores, min_number, max_number, true_number_community, label='Conductance')
    plotting_measure(generalized_conductance_scores, min_number, max_number, true_number_community, label='Generalized Conductance')
    plotting_markov_stability(generalized_markov_stability_scores, min_number, max_number, true_number_community)



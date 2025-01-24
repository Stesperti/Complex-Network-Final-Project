import numpy as np

def matrix_to_community_list(matrix):
    num_communities = matrix.shape[1]  # Number of columns = number of communities
    community_list = []
    
    for c in range(num_communities):
        # Find indices of nodes that belong to community c
        community_members = np.where(matrix[:, c] == 1)[0]
        community_list.append(list(community_members))
    
    return community_list


def select_maximum_row_community(matrix):
    """
    Selects the community with the highest value in each row of the given matrix
    and returns a binary matrix where 1 represents the selected community.
    """
    num_communities, num_features = matrix.shape
    community_matrix = np.zeros((num_communities, num_features))  # Use a tuple for the shape

    for i in range(num_communities):
        max_community = np.argmax(matrix[i])  # Find the index of the maximum value in the row
        community_matrix[i, max_community] = 1  # Set the corresponding position to 1

    community_list = matrix_to_community_list(community_matrix)

    community_list_new = [element for element in community_list if len(element)>0]

    return community_list_new, community_matrix
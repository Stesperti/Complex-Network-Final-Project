import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from metrix import norm_mutual_information

def plotting_measure(measure, min_number, max_number, true_community, label = 'Measure'):
    max_measure = np.max(measure)

    plt.figure()
    plt.title(label)
    plt.plot(range(min_number ,max_number),measure)
    plt.axvline(x = np.argmax(measure) + min_number , color = 'r', linestyle = '--', label = f'Maximum value at {np.argmax(measure) + min_number}')
    plt.axvline(x= true_community, color = 'g', linestyle = '--', label = f'True number of communities {true_community}')
    plt.xlabel('Number of communities')
    plt.ylabel(label)
    plt.legend()
    plt.show()  
    return measure


def plotting_markov_stability(measure, min_number, max_number, true_community):
    

    palette1 = sns.color_palette("tab10")  # A palette with 10 colors
    palette2 = sns.color_palette("Set1")  # Another palette, e.g., Set1 with 9 colors

    colors = palette1 + palette2

    if len(measure) > len(colors):
        colors = sns.color_palette("husl", n_colors=len(measure))  # Create a palette with as many colors as needed

    plt.figure()
    plt.title('Markov Stability')

    for idx, i in enumerate(measure):
        plt.plot(range(1, len(i) + 1), i, label=f'Partition {idx + min_number}', color=colors[idx])

    plt.xlabel('Time')
    plt.legend()
    plt.show()  
    return measure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from hiive.mdptoolbox.mdp import QLearning


def tree_QL(t, r, gamma_list, epsilon_list, alpha_list, alpha_decay_list):
    max_iterations = 1e10
    # to store results of each iteration
    results = pd.DataFrame(columns=['gamma', 'time', 'iterations', 'reward', 'error', 'max V', 'mean V', 'policy'])

    for gamma in gamma_list:
        for eps in epsilon_list:
            for alpha in alpha_list:
                for alpha_decay in alpha_decay_list:
                    print('Gamma: %s, Alpha: %s, Alpha Decay: %s, Epsilon Decay: %s'
                          % (str(gamma), str(alpha), str(alpha_decay), str(eps)))

                    test = QLearning(t, r, gamma=gamma, alpha=alpha, alpha_decay=alpha_decay,
                                     epsilon_decay=eps, n_iter=max_iterations)

                    runs = test.run()
                    results_list = [gamma, runs[-1]['Time'], runs[-1]['Iteration'], runs[-1]['Reward'],
                                    runs[-1]['Error'], runs[-1]['Max V'], runs[-1]['Mean V'], test.policy]
                    results.loc[len(results.index)] = results_list

                    print(results_list)
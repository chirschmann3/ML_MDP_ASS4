import gymnasium as gym
import hiive.mdptoolbox as mdptoolbox
import pandas as pd
from hiive.mdptoolbox.mdp import PolicyIteration
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


def tree_PI(t, r, gamma_list, error_gamma2plot, size):
    max_iterations = 100000
    # to store results of each iteration
    results = pd.DataFrame(columns=['gamma', 'time', 'iterations', 'reward', 'error', 'max V', 'mean V', 'policy'])

    for gamma in gamma_list:
        test = PolicyIteration(t, r, gamma, max_iter=max_iterations, eval_type='matrix')
        runs = test.run()
        results_list = [gamma, runs[-1]['Time'], runs[-1]['Iteration'], runs[-1]['Reward'],
                        runs[-1]['Error'], runs[-1]['Max V'], runs[-1]['Mean V'], test.policy]
        results.loc[len(results.index)] = results_list
        if gamma == error_gamma2plot: error2plot = [runs[i]['Error'] for i in range(len(runs))]

    # plot gamma vs Max and Mean V
    plt.plot(results['gamma'], results['max V'], label='Max Reward')
    plt.plot(results['gamma'], results['mean V'], label='Mean Reward')
    plt.title('Gamma vs Value')
    plt.legend()
    plt.xlabel('Gamma')
    plt.ylabel('Value')
    plt.savefig('images/PI/tree_valuegamma_' + str(size))
    plt.clf()

    # plot convergence of last iteration
    plt.plot(range(len(error2plot)), error2plot)
    plt.title('Error by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.savefig('images/PI/tree_convergence_' + str(size))
    plt.clf()

    # plot policy at 0.9 gamma
    sq_color = {0: 'k', 1: 'r'}
    sq_label = {0: 'Wt', 1: 'Ct'}
    policy = results[results['gamma'] == 0.9]['policy'].iloc[0]
    if len(policy) > 5:
        square_len = size // 5
        square_width = 5
        policy = np.array(list(policy)).reshape(square_width, square_len)
    else:
        square_len = 1
        square_width = len(policy)
        policy = np.array(policy)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlim=(-.01, square_width + 0.01), ylim=(-.01, square_len + 0.01))
    for i in range(square_width):
        for j in range(square_len):
            y = square_len - j - 1
            x = i
            pt = plt.Rectangle([x, y], 1, 1, edgecolor='k', linewidth=1)
            if square_len > 1:
                pt.set_facecolor(sq_color[policy[i, j]])
                ax.add_patch(pt)
                text = ax.text(x + 0.5, y + 0.5, sq_label[policy[i, j]],
                               horizontalalignment='center', size=10, verticalalignment='center', color='w')
            else:
                pt.set_facecolor(sq_color[policy[i]])
                ax.add_patch(pt)
                text = ax.text(x + 0.5, y + 0.5, sq_label[policy[i]],
                               horizontalalignment='center', size=10, verticalalignment='center', color='w')
    plt.title('Forest Clearing Policy')
    plt.axis('off')
    plt.savefig('images/PI/tree_policy_' + str(size))
    plt.clf()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from hiive.mdptoolbox.mdp import QLearning


def plot_value(hyperparam, df, size):
    plt.plot(df[hyperparam], df['max V'], label='Max Reward')
    plt.plot(df[hyperparam], df['mean V'], label='Mean Reward')
    plt.title('%s vs Value' + hyperparam)
    plt.legend()
    plt.xlabel(hyperparam)
    plt.ylabel('Value')
    plt.savefig('images/QLearn/tree_%s_valueby_%s.png' % (str(size), hyperparam))
    plt.clf()


def tree_QL(t, r, gamma_list, epsilon_list, alpha_list, alpha_decay_list, error_gamma2plot, size):
    max_iterations = 1e5
    # to store results of each iteration
    results = pd.DataFrame(columns=['gamma', 'epsilon', 'alpha', 'alpha decay',
                                    'time', 'iterations', 'reward', 'error', 'max V', 'mean V', 'policy'])

    # uncomment below to regather values but will take HOURS
    # instead use the CSV the results were written to
    for gamma in gamma_list:
        for eps in epsilon_list:
            for alpha in alpha_list:
                for alpha_decay in alpha_decay_list:
                    print('Gamma: %s, Alpha: %s, Alpha Decay: %s, Epsilon Decay: %s'
                          % (str(gamma), str(alpha), str(alpha_decay), str(eps)))

                    test = QLearning(t, r, gamma=gamma, alpha=alpha, alpha_decay=alpha_decay,
                                     epsilon_decay=eps, n_iter=max_iterations)

                    runs = test.run()
                    results_list = [gamma, eps, alpha, alpha_decay,
                                    runs[-1]['Time'], runs[-1]['Iteration'], runs[-1]['Reward'],
                                    runs[-1]['Error'], runs[-1]['Max V'], runs[-1]['Mean V'], test.policy]
                    results.loc[len(results.index)] = results_list
                    if gamma == error_gamma2plot:
                        error2plot = [runs[i]['Error'] for i in range(len(runs))]
                        values_list = [size, gamma, eps, alpha, alpha_decay]

    # write all results to CSV
    csvFile = 'images/QLearn/QL_results_%s.csv' % size
    results.to_csv(csvFile)

    # read CSV
    # COMMENT THIS OUT if re-running above
    # results = pd.read_csv('images/QLearn/QL_results.csv', index_col=0)

    # determine best combination
    best_combo_loc = results['max V'].idxmax()
    best_combo = results.iloc[best_combo_loc, :4]
    best_combo = list(zip(best_combo.index, best_combo))
    print('Highest Value Combination: ' + str(best_combo))

    # plot combos of other parameters for each optimal value
    params = ['gamma', 'epsilon', 'alpha', 'alpha decay']
    for param in params:
        steady_vals = [x for x in best_combo if x[0] != param]
        df = results[(results[steady_vals[0][0]] == steady_vals[0][1]) &
                     (results[steady_vals[1][0]] == steady_vals[1][1]) &
                     (results[steady_vals[2][0]] == steady_vals[2][1])]
        plot_value(param, df, size)

    # plot policy at optimal hyperparameters
    sq_color = {0: 'k', 1: 'r'}
    sq_label = {0: 'Wt', 1: 'Ct'}
    policy = results[(results[best_combo[0][0]] == best_combo[0][1]) &
                     (results[best_combo[1][0]] == best_combo[1][1]) &
                     (results[best_combo[2][0]] == best_combo[2][1]) &
                     (results[best_combo[3][0]] == best_combo[3][1])]['policy'].iloc[0]
    if isinstance(policy, str): policy = tuple(map(int, policy.strip('()').split(',')))
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
    plt.savefig('images/QLearn/tree_policy_%s.png' % str(size))
    plt.clf()

    # plot convergence of best combo
    test = QLearning(t, r, gamma=best_combo[0][1], alpha=best_combo[2][1], alpha_decay=best_combo[3][1],
                     epsilon_decay=best_combo[1][1], n_iter=max_iterations)

    runs = test.run()
    error = [runs[i]['Error'] for i in range(len(runs))]
    plt.plot(range(len(runs)), error)
    plt.title('Error by Iteration')
    plt.xlabel('Iteration')
    plt.ylim((0, 0.1))
    plt.ylabel('Error')
    plt.savefig('images/QLearn/tree_convergence_%s.png' % str(size))
    plt.clf()

    return results
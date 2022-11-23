import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

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


def tree_QL(t, r, gamma_list, epsilon_list, alpha_list, alpha_decay_list, size):
    max_iterations = 1e6
    # to store results of each iteration
    results = pd.DataFrame(columns=['gamma', 'epsilon', 'alpha', 'alpha decay',
                                    'time', 'iterations', 'reward', 'error', 'max V', 'mean V', 'policy'])

    # uncomment below to regather values but will take HOURS
    # instead use the CSV the results were written to
    # for gamma in gamma_list:
    #     for eps in epsilon_list:
    #         for alpha in alpha_list:
    #             for alpha_decay in alpha_decay_list:
    #                 print('Gamma: %s, Alpha: %s, Alpha Decay: %s, Epsilon Decay: %s'
    #                       % (str(gamma), str(alpha), str(alpha_decay), str(eps)))
    #
    #                 test = QLearning(t, r, gamma=gamma, alpha=alpha, alpha_decay=alpha_decay,
    #                                  epsilon_decay=eps, n_iter=max_iterations)
    #
    #                 runs = test.run()
    #                 results_list = [gamma, eps, alpha, alpha_decay,
    #                                 runs[-1]['Time'], runs[-1]['Iteration'], runs[-1]['Reward'],
    #                                 runs[-1]['Error'], runs[-1]['Max V'], runs[-1]['Mean V'], test.policy]
    #                 results.loc[len(results.index)] = results_list
    #
    # # write all results to CSV
    # csvFile = 'images/QLearn/QL_results_%s.csv' % size
    # results.to_csv(csvFile)

    # read CSV
    # COMMENT THIS OUT if re-running above
    filename = 'images/QLearn/QL_results_%s.csv' % str(size)
    results = pd.read_csv(filename, index_col=0)

    # determine best combination
    best_combo_loc = results['max V'].idxmax()
    best_combo = results.iloc[best_combo_loc, :4]
    best_combo = list(zip(best_combo.index, best_combo))
    print('Highest Value Combination: ' + str(best_combo))

    # plot combos of other parameters for each optimal value
    print('Plotting things')
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
                     epsilon_decay=best_combo[1][1], n_iter=1e8)

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


def lake_QL(env, gamma, eps_decay, alpha, alpha_decay):
    # code adapted from https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2
    eps = 1.0
    eps_min = 0.1
    alpha_min = 0.001
    theta = 1e-15 # value to break if Q isn't changing
    max_episodes = 1e5
    rev_list = []
    error_list = []
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    start = time.time()
    for i in range(int(max_episodes)):
        s = env.reset()[0]
        rALL = 0
        d = False
        j = 0
        error = []
        # former_q = Q
        # run episode up to 5000 steps
        while j < 500:
            # env.render()
            j+=1
            # action choice: greedy with increasing probability
            # random action for epsilon and greedy with 1-eps
            pn = np.random.random()
            if pn < eps:
                a = np.random.randint(0, env.action_space.n)
            else:
                # optimal action
                a = Q[s, :].argmax()
            # get new state and reward
            s1, r, d, _, _ = env.step(a)
            # update Q-Table
            dQ = alpha * (r + gamma * np.max(Q[s1,:]) - Q[s,a])
            Q[s,a] = Q[s,a] + dQ
            error.append(np.absolute(dQ))
            # decay alpha and epsilon
            alpha *= alpha_decay
            if alpha < alpha_min: alpha = alpha_min
            eps *= eps_decay
            if eps < eps_min: eps = eps_min
            rALL += r
            s = s1
            if d:
                break

        rev_list.append(rALL)
        # error is looked at as the mean update to the Q table at each move in the episode
        error_list.append(np.mean(error))
        # env.render()

        # break if all differences are smaller than theta
        # if Q.sum() > 0 & np.all((former_q - Q) < theta):
        #     break

    total_time = time.time() - start
    avg_reward = sum(rev_list)/max_episodes
    policy = np.argmax(Q, axis=1)
    return [total_time, avg_reward, error_list, policy]


def lake_QL_experiments(env, gamma_list, epsilon_list, alpha_list, alpha_decay_list):
    # to store results of each iteration
    results = pd.DataFrame(columns=['gamma', 'epsilon', 'alpha', 'alpha decay',
                                    'time', 'reward', 'error', 'policy'])

    # uncomment below to regather values but will take HOURS
    # instead use the CSV the results were written to
    for gamma in gamma_list:
        for eps_decay in epsilon_list:
            for alpha in alpha_list:
                for alpha_decay in alpha_decay_list:
                    print('Gamma: %s, Alpha: %s, Alpha Decay: %s, Epsilon Decay: %s'
                          % (str(gamma), str(alpha), str(alpha_decay), str(eps_decay)))

                    run = lake_QL(env, gamma, eps_decay, alpha, alpha_decay)

                    results_list = [gamma, eps_decay, alpha, alpha_decay,
                                    run[0], run[1], run[2], run[3]]
                    results.loc[len(results.index)] = results_list

    # write all results to CSV
    csvFile = 'images/QLearn/QL_results_lake.csv'
    results.to_csv(csvFile)

    # read from CSV
    # comment out if rerunning above
    results = pd.read_csv('images/QLearn/QL_results_lake.csv', index_col=0)

    # get optimal run
    best_combo_loc = results['reward'].idxmax()
    best_combo = results.iloc[best_combo_loc, :4]
    best_combo = list(zip(best_combo.index, best_combo))
    print('Highest Value Combination: ' + str(best_combo))

    # plot policy from optimal run
    best_policy = results['policy'].iloc(best_combo_loc)
    marked_policy(best_policy, env)

    # plot combos of other parameters for each optimal value
    print('Plotting things')
    params = ['gamma', 'epsilon', 'alpha', 'alpha decay']
    for param in params:
        steady_vals = [x for x in best_combo if x[0] != param]
        df = results[(results[steady_vals[0][0]] == steady_vals[0][1]) &
                     (results[steady_vals[1][0]] == steady_vals[1][1]) &
                     (results[steady_vals[2][0]] == steady_vals[2][1])]
        plot_value_lake(param, df)

    # plot convergence of best combo
    error = results['error'].iloc(best_combo_loc)
    plt.plot(range(len(error)), error)
    plt.title('Avg QTable Update by Episode')
    plt.xlabel('Episode')
    # plt.ylim((0, 0.1))
    plt.ylabel('Avg Change')
    plt.savefig('images/QLearn/FrozenLake_convergence.png')
    plt.clf()


def marked_policy(P, env):
    # https://towardsdatascience.com/this-is-how-reinforcement-learning-works-5080b3a335d6
    # function for displaying a heatmap
    nb_states = env.observation_space.n
    actions = P
    state_labels = np.where(actions == 0, '<',
                            np.where(actions == 1, 'v',
                                     np.where(actions == 2, '>',
                                              np.where(actions == 3, '^', 0)
                                              )
                                     )
                            )
    desc = env.unwrapped.desc.ravel().astype(str)
    color_values = np.where(desc == 'S', 0,
                      np.where(desc == 'F', 1,
                               np.where(desc == 'H', 2,
                                        np.where(desc == 'G', 3, desc))))
    colors = np.where(desc == 'S', 'y',
                      np.where(desc == 'F', 'b',
                               np.where(desc == 'H', 'r',
                                        np.where(desc == 'G', 'g', desc))))
    ax = sns.heatmap(color_values.astype(int).reshape(int(np.sqrt(nb_states)), int(np.sqrt(nb_states))),
                     linewidth=0.5,
                     annot=state_labels.reshape(int(np.sqrt(nb_states)), int(np.sqrt(nb_states))),
                     cmap=list(colors),
                     fmt='',
                     cbar=False)
    # if want colors to correspond with moves, change color_values to np.argmax(P,axis=1).reshape...
    plt.savefig('images/QLearn/FrozenLake_policy.png')
    plt.clf()


def plot_value_lake(hyperparam, df):
    plt.plot(df[hyperparam], df['reward'], label='Avg Episode Reward')
    plt.title('%s vs Value' + hyperparam)
    plt.legend()
    plt.xlabel(hyperparam)
    plt.ylabel('Value')
    plt.savefig('images/QLearn/FrozenLake_valueby_%s.png' % hyperparam)
    plt.clf()

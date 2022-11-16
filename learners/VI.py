"""
Value Iteration as provided by:
https://towardsdatascience.com/value-iteration-to-solve-openai-gyms-frozenlake-6c5e7bf0a64d

"""

import gymnasium as gym
import hiive.mdptoolbox as mdptoolbox
import pandas as pd
from hiive.mdptoolbox.mdp import ValueIteration
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_convergence(delta_list, env_name, gamma):
    x = range(len(delta_list))
    plt.plot(x, delta_list)
    plt.title('Delta by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Delta')
    plt.savefig('images/%s_convergence_%s.png' % (env_name, str(gamma)))
    plt.clf()


def argmax(env, V, pi, action, s, gamma):
    e = np.zeros(env.action_space.n)
    for a in range(env.action_space.n):  # iterate for every action possible
        q = 0
        P = np.array(env.env.P[s][a])
        (x, y) = np.shape(P)  # for Bellman Equation

        for i in range(x):  # iterate for every possible states
            s_ = int(P[i][1])  # S' - Sprime - possible succesor states
            p = P[i][0]  # Transition Probability P(s'|s,a)
            r = P[i][2]  # Reward

            q += p * (r + gamma * V[s_])  # calculate action_ value q(s|a)
            e[a] = q

    m = np.argmax(e)
    action[s] = m  # Take index which has maximum value
    pi[s][m] = 1  # update pi(a|s)

    return pi


def bellman_optimality_update(env, V, s, gamma):  # update the stae_value V[s] by taking
    pi = np.zeros((env.observation_space.n, env.action_space.n))  # action which maximizes current value
    e = np.zeros(env.action_space.n)
    # STEP1: Find
    for a in range(env.action_space.n):
        q = 0  # iterate for all possible action
        P = np.array(env.env.P[s][a])
        (x, y) = np.shape(P)

        for i in range(x):
            s_ = int(P[i][1])
            p = P[i][0]
            r = P[i][2]
            q += p * (r + gamma * V[s_])
            e[a] = q

    m = np.argmax(e)
    pi[s][m] = 1

    value = 0
    for a in range(env.action_space.n):
        u = 0
        P = np.array(env.env.P[s][a])
        (x, y) = np.shape(P)
        for i in range(x):
            s_ = int(P[i][1])
            p = P[i][0]
            r = P[i][2]

            u += p * (r + gamma * V[s_])

        value += pi[s, a] * u

    V[s] = value
    return V[s]


def value_iteration(env, gamma, theta, env_name):
    V = np.zeros(env.observation_space.n)  # initialize v(0) to arbitory value, my case "zeros"
    delta_list = []
    while True:
        delta = 0
        for s in range(env.observation_space.n):  # iterate for all states
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)  # update state_value with bellman_optimality_update
            delta = max(delta, abs(v - V[s]))  # assign the change in value per iteration to delta
            delta_list.append(delta) # tracks the deltas each iteration so we can plot convergence
        if delta < theta:
            break  # if change gets to negligible
            # --> converged to optimal value
    pi = np.zeros((env.observation_space.n, env.action_space.n))
    action = np.zeros((env.observation_space.n))
    for s in range(env.observation_space.n):
        pi = argmax(env, V, pi, action, s, gamma)  # extract optimal policy using action value

    # plot convergence
    plot_convergence(delta_list, env_name, gamma)

    return V, pi, action  # optimal value funtion, optimal policy


def plot_value(V, env_name, gamma):
    x = range(V.shape[0])
    plt.bar(x, V)
    plt.title('Value Over States')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.savefig('images/%s_values_%s.png' % (env_name, str(gamma)))
    plt.clf()


def plot_policy(pi, env_name, gamma):
    sns.set()
    ax = sns.heatmap(pi.transpose(), yticklabels=['Left', 'Down', 'Up', 'Right'])
    plt.savefig('images/%s_policy_%s.png' % (env_name, str(gamma)))
    plt.clf()


def marked_policy(P, env, env_name, gamma):
    # https://towardsdatascience.com/this-is-how-reinforcement-learning-works-5080b3a335d6
    # function for displaying a heatmap
    nb_states = env.observation_space.n
    actions = np.argmax(P, axis=1)
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
    plt.savefig('images/%s_movesmade_%s.png' % (env_name, str(gamma)))
    plt.clf()


def tree_VI(t, r, gamma_list, theta, size):
    max_iterations = 100000
    results = pd.DataFrame(columns=['gamma', 'time', 'iterations', 'reward', 'error', 'max V', 'mean V', 'policy'])

    for gamma in gamma_list:
        test = ValueIteration(t, r, gamma=gamma, epsilon=theta, max_iter=max_iterations)
        runs = test.run()
        results_list = [gamma, runs[-1]['Time'], runs[-1]['Iteration'], runs[-1]['Reward'],
                        runs[-1]['Error'], runs[-1]['Max V'], runs[-1]['Mean V'], test.policy]
        results.loc[len(results.index)] = results_list

    # plot gamma vs Max and Mean V
    plt.plot(results['gamma'], results['max V'], label='Max V')
    plt.plot(results['gamma'], results['mean V'], label='Mean V')
    plt.title('Gamma vs Value')
    plt.legend()
    plt.xlabel('Gamma')
    plt.ylabel('Value')
    plt.savefig('images/tree_valuegamma_' + str(size))
    plt.clf()

    # plot convergence of last iteration
    error = [runs[i]['Error'] for i in range(len(runs))]
    plt.plot(range(len(error)), error)
    plt.title('Error by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.savefig('images/tree_convergence_' + str(size))
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
    ax = fig.add_subplot(111, xlim=(-.01, square_width+0.01), ylim=(-.01, square_len+0.01))
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
                text = ax.text(x+0.5, y+0.5, sq_label[policy[i]],
                               horizontalalignment='center', size=10, verticalalignment='center', color='w')
    plt.title('Forest Clearing Policy')
    plt.axis('off')
    plt.savefig('images/tree_policy_' + str(size))
    plt.clf()


    return results




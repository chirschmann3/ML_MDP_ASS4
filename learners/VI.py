"""
Value Iteration as provided by:
https://towardsdatascience.com/value-iteration-to-solve-openai-gyms-frozenlake-6c5e7bf0a64d

"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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


def value_iteration(env, gamma, theta):
    V = np.zeros(env.observation_space.n)  # initialize v(0) to arbitory value, my case "zeros"
    while True:
        delta = 0
        for s in range(env.observation_space.n):  # iterate for all states
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)  # update state_value with bellman_optimality_update
            delta = max(delta, abs(v - V[s]))  # assign the change in value per iteration to delta
        if delta < theta:
            break  # if change gets to negligible
            # --> converged to optimal value
    pi = np.zeros((env.observation_space.n, env.action_space.n))
    action = np.zeros((env.observation_space.n))
    for s in range(env.observation_space.n):
        pi = argmax(env, V, pi, action, s, gamma)  # extract optimal policy using action value

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


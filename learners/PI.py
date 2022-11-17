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


# below from
# https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
def run_episode(env, policy, gamma=1.0, render=False):
    """ Runs an episode and return the total reward """
    max_attempts = 10000
    obs = env.reset()[0]
    total_reward = 0
    step_idx = 0
    reached_end = False
    while True and max_attempts > 0:
        if render:
            env.render()
        obs, reward, done, _, _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        max_attempts -= 1
        if done:
            reached_end = True
            break
    return total_reward, reached_end


def evaluate_policy(env, policy, gamma=1.0, n=100):
    outcome = [run_episode(env, policy, gamma, False) for _ in range(n)]
    count_success = sum(bool(x) for _, x in outcome)
    rewards = [x for x, _ in outcome]
    max_r = max(rewards)
    mean_r = np.mean(rewards)
    return count_success, max_r, mean_r


def extract_policy(env, v, gamma=1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy


def compute_policy_v(env, policy, eps=1e-12, gamma=1.0):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s]
    and solve them to find the value function.
    """
    v = np.zeros(env.observation_space.n)
    while True:
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v


def policy_iteration(env, gamma=1.0, eps=1e-12):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n))  # initialize a random policy
    max_iterations = 200000
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, eps, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print('Policy-Iteration converged at step %d.' % (i+1))
            break
        policy = new_policy
    return policy

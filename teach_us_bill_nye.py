"""
Main place to gather and learn from the all knowledgable Bill Nye.
I.e. run all experiments here....
"""
import random

from learners import VI, PI, qlearner
import gymnasium as gym
import hiive.mdptoolbox.example
import numpy as np
import time
import random


# def test_gamma(gamma_list, env_name, iteration_type):
#     for gamma in gamma_list:
#         print('gamma = %s' % str(gamma))
#         env = gym.make(env_name)
#         if iteration_type == 'value_iteration':
#             start = time.time()
#             optimal_v, env = value_iteration.value_iteration(env, gamma)
#             print('Time to converge: ' + str((time.time() - start)))
#             policy = value_iteration.extract_policy(optimal_v, env, gamma)
#             policy_scores = value_iteration.evaluate_policy(env, policy, gamma, n=1000)
#             print('Policy avg score = %s' % np.mean(policy_scores))


def run_value_iteration(gamma_list, theta, env_name, *args):
    if env_name == 'FrozenLake8x8-v1':
        env = gym.make(env_name)
        print(env.desc)
        for gamma in gamma_list:
            print('gamma = %s' % str(gamma))
            start = time.time()
            V, pi, action = VI.value_iteration(env, gamma, theta, env_name)
            print('Time to converge: ' + str((time.time() - start)))

            a = np.reshape(action, (env.nrow, env.ncol))
            print('Policy Actions to Take')
            print(a)  # discrete action to take in given state

            VI.plot_value(V, env_name, gamma)
            VI.plot_policy(pi, env_name, gamma)
            VI.marked_policy(pi, env=env, env_name=env_name, gamma=gamma)

            e = 0
            for i_episode in range(100):
                c = env.reset()[0]
                for t in range(10000):
                    c, reward, done, info, _ = env.step(action[c])
                    if done:
                        if reward == 1:
                            e += 1
                        break
            print(" agent succeeded to reach goal {} out of 100 Episodes using this policy ".format(e))
            print()
    elif env_name == 'Forest':
        for s in args[0]:
            T, R = hiive.mdptoolbox.example.forest(S=s)
            results = VI.tree_VI(T, R, gamma_list, theta, s)
            # print('Results by Gamma:')
            # print(results[['gamma', 'time', 'iterations', 'reward', 'error', 'max V', 'mean V']])
            # print('Policy Highest Gamma')
            # print(results['policy'].iloc[-1])


def run_policy_iteration(gamma_list, eps, env_name, error_gamma2plot, *args):
    if env_name == 'Forest':
        for s in args[0]:
            T, R = hiive.mdptoolbox.example.forest(S=s)
            results = PI.tree_PI(T, R, gamma_list, error_gamma2plot, s)
            # print('Results by Gamma:')
            # print(results[['gamma', 'time', 'iterations', 'reward', 'error', 'max V', 'mean V']])
            # print('Policy Highest Gamma')
            # print(results['policy'].iloc[-1])
    else:
        env = gym.make(env_name)
        print(env.desc)
        for gamma in gamma_list:
            print('gamma = %s' % str(gamma))
            start = time.time()
            pi = PI.policy_iteration(env, env_name, gamma, eps)
            print('Time to converge: ' + str((time.time() - start)))
            num_success, max_r, mean_r = PI.evaluate_policy(env, pi, gamma)
            print(" agent succeeded to reach goal {} out of 100 Episodes using this policy ".format(num_success))
            a = np.reshape(pi, (env.nrow, env.ncol))
            print('Policy Actions to Take')
            print(a)  # discrete action to take in given state
            print()

            PI.marked_policy(pi, env=env, env_name=env_name, gamma=gamma)


def run_qlearning(env_name, gamma_list, epsilon_list, alpha_list, alpha_decay_list, error_gamma2plot, *args):
    if env_name == 'Forest':
        for s in args[0]:
            print('Checking state size: ' + str(s))
            T, R = hiive.mdptoolbox.example.forest(S=s)
            results = qlearner.tree_QL(T, R, gamma_list, epsilon_list,
                                       alpha_list, alpha_decay_list, error_gamma2plot, s)
    else:
        env = gym.make(env_name)



if __name__=='__main__':
    np.random.seed(9)
    random.seed(9)

    # run frozen lake value iteration
    # env_name = 'FrozenLake8x8-v1'
    # gamma_list = [0.0001, 0.001, 0.1, 0.25, 0.5, 0.75, 1.0]
    # eps = 1e-12
    # print('\nRunning Frozen Lake Value Iteration\n')
    # run_value_iteration(gamma_list, eps, env_name)
    # print('\nRunning Frozen Lake Policy Iteration\n')
    # run_policy_iteration(gamma_list, eps, env_name, 0.8)

    # run forest experiments
    env_name = 'Forest'
    # gamma_list = [0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
    # state_sizes = [3, 5, 10, 50, 100, 500]
    # print('\nRunning Forest Value Iteration\n')
    # run_value_iteration(gamma_list, eps, env_name, state_sizes)
    # print('\nRunning Forest Policy Iteration\n')
    # gamma_list = [0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    # run_policy_iteration(gamma_list, eps, env_name, 0.8, state_sizes)
    print('\nRunning Forest QLearning\n')
    gamma_list = [0.1, 0.5, 0.8, 0.9, 0.99]
    epsilon_decay_list = [0.1, 0.5, 0.9, 0.99]
    alpha_list = [0.1, 0.5, 0.9, 0.99]
    alpha_decay = [0.1, 0.5, 0.9, 0.99]
    state_size = [100]
    run_qlearning(env_name, gamma_list, epsilon_decay_list, alpha_list, alpha_decay, 0.8, state_size)

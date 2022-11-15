"""
Main place to gather and learn from the all knowledgable Bill Nye.
I.e. run all experiments here....
"""

from learners import value_iteration
import gymnasium as gym
import numpy as np

def test_gamma(gamma_list, env_name, iteration_type):
    for gamma in gamma_list:
        print('gamma = %s' % str(gamma))
        env = gym.make(env_name)
        if iteration_type=='value_iteration':
            optimal_v = value_iteration.value_iteration(env, gamma)
            policy = value_iteration.extract_policy(optimal_v, gamma)
            policy_scores = value_iteration.evaluate_policy(env, policy, gamma, n=1000)
            print('Policy avg score = %s' % np.mean(policy_scores))


if __name__=='__main__':
    np.random.seed(9)

    # run frozen lake experiments
    env_name = 'FrozenLake8x8-v0'
    print('\nRunning Frozen Lake Experiments\n')
    test_gamma([0.000001, 0.001, 0.1, 1.0], env_name, 'value_iteration')
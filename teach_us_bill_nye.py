"""
Main place to gather and learn from the all knowledgable Bill Nye.
I.e. run all experiments here....
"""

from learners import value_iteration, VI
import gymnasium as gym
import numpy as np
import time


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


def tg(gamma_list, env_name, iteration_type):
    env = gym.make(env_name)
    print(env.desc)
    for gamma in gamma_list:
        print('gamma = %s' % str(gamma))
        if iteration_type == 'value_iteration':
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
            print(" agent succeeded to reach goal {} out of 100 Episodes using this policy ".format(e + 1))
            print()
            # env.close()


if __name__=='__main__':
    np.random.seed(9)

    # run frozen lake experiments
    env_name = 'FrozenLake8x8-v1'
    print('\nRunning Frozen Lake Experiments\n')
    gamma_list = [0.0001, 0.001, 0.1, 0.5, 1.0]
    theta = 0.000001
    print('\nRunning Frozen Lake Experiments\n')
    tg(gamma_list, env_name, 'value_iteration')
import numpy as np
import torch

def epsilon_greedy(env, epsilon, Q):
    """
    Chooses an epsilon-greedy action starting from a given state and given Q-values
    :param env: environment
    :param epsilon: current exploration parameter
    :param Q: current Q-values.
    :return:
        - the chosen action
    """

    if np.random.uniform(0, 1) < epsilon:
        # with probability epsilon make a random move (exploration)
        return env.action_space.sample()
    else:
        # with probability 1-epsilon choose the action with the highest immediate reward (exploitation)
        return torch.argmax(Q)
import numpy as np


def epsilon_greedy(env, epsilon, Q):
    """
    Chooses an epsilon-greedy action starting from a given state and given Q-values
    :param env: environment
    :param epsilon: current exploration parameter
    :param Q: current Q-values.
    :return:
        - the chosen action
    """
    # get the available positions
    available_actions = env.available()

    if np.random.uniform(0, 1) < epsilon:
        # with probability epsilon make a random move (exploration)
        return str(np.random.choice(available_actions))
    else:
        # with probability 1-epsilon choose the action with the highest immediate reward (exploitation)
        q = np.copy(Q[env.get_state()])
        mask = [env.encode_action(action) for action in available_actions]
        q = [q[i] if i in mask else np.nan for i in range(len(q))]
        max_indices = np.argwhere(q == np.nanmax(q)).flatten()  # best action(s) along the available ones
        return env.inverse_encoding(int(np.random.choice(max_indices)))  # ties are split randomly
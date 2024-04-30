from collections import defaultdict, deque

import numpy as np
from scipy.special import softmax


def softmax_(env, beta, Q):
    """
    Chooses an action using softmax distribution over the available actions
    :param env: environment
    :param beta: scaling parameter of the softmax policy
    :param Q: current Q-values
    :return:
        - the chosen action
    """
    # get the available positions
    available_actions = env.available()

    q = np.copy(Q[env.get_state()])
    # Apply the scaling parameter to the action choices
    q = beta * q
    mask = [env.encode_action(action) for action in available_actions]
    return str(np.random.choice(available_actions, p=softmax(q[mask])))


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


def q_learning(env, alpha=0.05, gamma=0.99, num_episodes=1000, action_policy="epsilon_greedy", epsilon_exploration=0.1,
               epsilon_exploration_rule=None, trace_decay=0, initial_q=0):
    """
    Trains an agent using the Q-Learning algorithm, by playing num_episodes until the end.
    :param env: the environment
    :param alpha: learning rate
    :param gamma: discount rate for future rewards
    :param num_episodes: number of training episodes
    :param action_policy: string for the action policy to be followed during training
        It is usually "epsilon_greedy" or "softmax_"
    :param trace_decay: trace decay factor for eligibility traces
        If 0, q_learning(0) is implemented without any eligibility trace. If a non-zero float is given in input
        the latter represents the trace decay factor and q_learning(lambda) is implemented
    :param epsilon_exploration: parameter of the exploration policy: exploration rate or softmax_ scaling factor
        If action_policy is "epsilon_greedy":
            If epsilon_exploration_rule is None, at each iteration the action with the highest Q-value
            is taken with probability (1-epsilon_exploration)
        If action_policy is "softmax_":
            epsilon_exploration is actually beta, the scaling factor for the softmax.
    :param epsilon_exploration_rule: function mapping each positive integer to the exploitation epsilon
        of the corresponding episode.
        If epsilon_exploration_rule is not None, at episode number n during training the parameter for the 
        exploration policy is epsilon_exploration_rule(n).
    :param initial_q: initialization value of all Q-values
    :return:
        - Q: empirical estimates of the Q-values
        - stats: dictionary of statistics collected during training, with keys 'episode_rewards' and 'episode_lengths',
            and the corresponding values being a list (of length num_episodes) containing, for each episode, the reward
            collected and the length of the episode respectively.
    """

    # Q-values map
    # Dictionary that maps the tuple representation of the state to a dictionary of action values
    Q = defaultdict(lambda: initial_q * np.ones(env.get_num_actions()))  # All Q-values are initialized to initial_q
    # Stats of training
    episode_rewards = np.empty(num_episodes)
    episode_lengths = np.empty(num_episodes)

    assert action_policy in ("epsilon_greedy", "softmax_")
    # Set the default value of the exploration parameter
    if epsilon_exploration is None:
        if action_policy == "epsilon_greedy":
            epsilon_exploration = 0.1
        else:
            epsilon_exploration = 2

    if epsilon_exploration_rule is None:
        def epsilon_exploration_rule(n):
            return epsilon_exploration  # if an exploration rule is not given, it is the constant one

    for itr in range(num_episodes):
        env.reset()
        length = 0
        total_reward = 0
        e = defaultdict(lambda: np.zeros(env.get_num_actions()))  # shadow variable for each state action pair, all to 0

        # first state outside the loop
        state = env.get_state()

        while not env.end:

            # rescale all traces
            for key in list(e):
                e[key] *= trace_decay

            # choose action according to the desired policy
            action = eval(action_policy)(env, epsilon_exploration_rule(itr + 1), Q)
            next_state, reward = env.do_action(action)  # Move according to the policy

            length += 1
            total_reward += reward

            # current state action pair
            e[state][env.encode_action(action)] += 1

            if not env.end:
                next_greedy_action = epsilon_greedy(env, 0, Q=Q)
                target = reward + gamma * Q[next_state][env.encode_action(next_greedy_action)]
            else:
                target = reward  # the fictitious Q-value of Q(next_state)[\cdot] is zero

            # update all Q-values
            for key in list(e):
                Q[key] += alpha * (target - Q[state][env.encode_action(action)]) * e[key]

            # Preparing for the next move
            state = next_state

        episode_rewards[itr] = total_reward  # reward of the current episode
        episode_lengths[itr] = length  # length of the current episode
    # Dictionary of stats
    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }

    return Q, stats


def sarsa(env, alpha=0.05, gamma=0.99, num_episodes=1000, action_policy="epsilon_greedy", epsilon_exploration=0.1,
          epsilon_exploration_rule=None, trace_decay=0, initial_q=0):
    """
    Trains an agent using the Sarsa algorithm, by playing num_episodes until the end.
    :param env: the environment
    :param alpha: learning rate
    :param gamma: discount rate for future rewards
    :param num_episodes: number of training episodes
    :param action_policy: string for the action policy to be followed during training
        It is usually "epsilon_greedy" or "softmax_"
    :param trace_decay: trace decay factor for eligibility traces
        If 0, sarsa(0) is implemented without any eligibility trace. If a non-zero float is given in input
        the latter represents the trace decay factor and sarsa(lambda) is implemented
    :param epsilon_exploration: parameter of the exploration policy: exploration rate or softmax_ scaling factor
        If action_policy is "epsilon_greedy":
            If epsilon_exploration_rule is None, at each iteration the action with the highest Q-value
            is taken with probability (1-epsilon_exploration)
        If action_policy is "softmax_":
            epsilon_exploration is actually beta, the scaling factor for the softmax.
    :param epsilon_exploration_rule: function mapping each positive integer to the exploitation epsilon
        of the corresponding episode.
        If epsilon_exploration_rule is not None, at episode number n during training the action
        with the highest Q-value is taken with probability (1-epsilon_exploration_rule(n))
    :param initial_q: initialization value of all Q-values
    :return:
        - Q: empirical estimates of the Q-values
        - stats: dictionary of statistics collected during training, with keys 'episode_rewards' and 'episode_lengths',
            and the corresponding values being a list (of length num_episodes) containing, for each episode, the reward
            collected and the length of the episode respectively.
    """
    # Q-values map
    # Dictionary that maps the tuple representation of the state to a dictionary of action values
    Q = defaultdict(lambda: initial_q * np.ones(env.get_num_actions()))  # All Q-values are initialized to initial_q
    # Stats of training
    episode_rewards = np.empty(num_episodes)
    episode_lengths = np.empty(num_episodes)

    assert action_policy in ("epsilon_greedy", "softmax_")

    if epsilon_exploration_rule is None:
        def epsilon_exploration_rule(n):
            return epsilon_exploration  # if an exploration rule is not given, it is the constant one

    for itr in range(num_episodes):
        env.reset()
        length = 0
        total_reward = 0
        e = defaultdict(lambda: np.zeros(env.get_num_actions()))  # shadow variable for each state action pair, all to 0

        # first state and action outside the loop
        state = env.get_state()

        # choose action according to the desired policy
        action = eval(action_policy)(env, epsilon_exploration_rule(itr + 1), Q=Q)

        while not env.end:

            # rescale all traces
            for key in list(e):
                e[key] *= trace_decay

            # Move according to the policy
            next_state, reward = env.do_action(action)
            length += 1
            total_reward += reward

            # current state action pair
            e[state][env.encode_action(action)] += 1

            if not env.end:
                # choose action according to the desired policy
                next_action = eval(action_policy)(env, epsilon_exploration_rule(itr + 1), Q=Q)
                target = reward + gamma * Q[next_state][env.encode_action(next_action)]
            else:
                target = reward  # the fictitious Q-value of Q(next_state)[\cdot] is zero

            # update all Q-values
            for key in list(e):
                Q[key] += alpha * (target - Q[state][env.encode_action(action)]) * e[key]

            # Preparing for the next move
            state = next_state

            if not env.end:
                action = next_action

        episode_rewards[itr] = total_reward  # reward of the current episode
        episode_lengths[itr] = length  # length of the current episode

    # Dictionary of stats
    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }

    return Q, stats


def n_step_sarsa(env, alpha=0.05, gamma=0.99, num_episodes=1000, action_policy="epsilon_greedy", n=1,
                 epsilon_exploration=0.5, epsilon_exploration_rule=None, initial_q=0):
    """
    Trains an agent using the Sarsa algorithm, by playing num_episodes until the end.
    :param env: the environment
    :param alpha: learning rate
    :param gamma: discount rate for future rewards
    :param num_episodes: number of training episodes
    :param action_policy: string for the action policy to be followed during training
        It is usually "epsilon_greedy" or "softmax_"
    :param n: for n = 1 standard Sarsa(0) is recovered, otherwise n-step Sarsa is implemented
    :param epsilon_exploration: parameter of the exploration policy: exploration rate or softmax_ scaling factor
        If action_policy is "epsilon_greedy":
            If epsilon_exploration_rule is None, at each iteration the action with the highest Q-value
            is taken with probability (1-epsilon_exploration)
        If action_policy is "softmax_":
            epsilon_exploration is actually beta, the scaling factor for the softmax.
    :param epsilon_exploration_rule: function mapping each positive integer to the exploitation epsilon
        of the corresponding episode.
        If epsilon_exploration_rule is not None, at episode number n during training the action
        with the highest Q-value is taken with probability (1-epsilon_exploration_rule(n))
    :param initial_q: initialization value of all Q-values
    :return:
        - Q: empirical estimates of the Q-values
        - stats: dictionary of statistics collected during training, with keys 'episode_rewards' and 'episode_lengths',
            and the corresponding values being a list (of length num_episodes) containing, for each episode, the reward
            collected and the length of the episode respectively.
    """
    # Q-values map
    # Dictionary that maps the tuple representation of the state to a dictionary of action values
    Q = defaultdict(lambda: initial_q * np.ones(env.get_num_actions()))  # All Q-values are initialized to initial_q
    # Stats of training
    episode_rewards = np.empty(num_episodes)
    episode_lengths = np.empty(num_episodes)

    assert action_policy in ("epsilon_greedy", "softmax_")

    if epsilon_exploration_rule is None:
        def epsilon_exploration_rule(n):
            return epsilon_exploration  # if an exploration rule is not given, it is the constant one

    reward_weights = np.array([gamma ** i for i in range(n)])

    for itr in range(num_episodes):
        env.reset()
        length = 0
        total_reward = 0

        # first state and action outside the loop
        state = env.get_state()

        current_episode_states_actions = deque(maxlen=n+1)
        current_episode_rewards = deque(maxlen=n)

        # choose action according to the desired policy
        action = eval(action_policy)(env, epsilon_exploration_rule(itr + 1), Q=Q)
        current_episode_states_actions.append((state, action))
        next_action = None

        while not env.end:

            # Move according to the policy
            next_state, reward = env.do_action(action)
            current_episode_rewards.append(reward)

            length += 1
            total_reward += reward

            if not env.end:
                next_action = eval(action_policy)(env, epsilon_exploration_rule(itr + 1), Q=Q)
                current_episode_states_actions.append((next_state, next_action))
                target = np.dot(np.array(current_episode_rewards), reward_weights[:len(current_episode_rewards)]) + \
                    gamma ** n * Q[next_state][env.encode_action(next_action)]
            else:
                target = np.dot(np.array(current_episode_rewards), reward_weights[:len(current_episode_rewards)])

            if len(current_episode_rewards) == current_episode_rewards.maxlen and not env.end:
                Q[current_episode_states_actions[0][0]][env.encode_action(current_episode_states_actions[0][1])] += \
                    alpha * (target - Q[current_episode_states_actions[0][0]][env.encode_action(current_episode_states_actions[0][1])])

            action = next_action

        # update remaining Q-values within n steps from the reward
        current_episode_states_actions.popleft()

        for i in range(len(current_episode_rewards)):
            target = np.dot(np.array(current_episode_rewards)[i:], reward_weights[:(len(current_episode_rewards)-i)])
            current_element = current_episode_states_actions[i]
            Q[current_element[0]][env.encode_action(current_element[1])] += \
                alpha * (target - Q[current_element[0]][env.encode_action(current_element[1])])

        episode_rewards[itr] = total_reward  # reward of the current episode
        episode_lengths[itr] = length  # length of the current episode

    # Dictionary of stats
    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }

    return Q, stats

####################### NOT USED #######################
def td(env, alpha=0.05, gamma=0.99, num_episodes=1000, epsilon_exploration=1, action_policy="epsilon_greedy",
       epsilon_exploration_rule=None, trace_decay=0, initial_v=0):
    """
    Trains an agent using the TD algorithm, by playing num_episodes until the end.
    :param env: the environment
    :param alpha: learning rate
    :param gamma: discount rate for future rewards
    :param num_episodes: number of training episodes
    :param action_policy: string for the action policy to be followed during training
    :param trace_decay: trace decay factor for eligibility traces
        If 0, TD(0) is implemented without any eligibility trace. If a non-zero float is given in input
        the latter represents the trace decay factor and TD(lambda) is implemented
    :param epsilon_exploration: exploration rate of the action_policy
    :param epsilon_exploration_rule: function mapping each positive integer to the exploitation epsilon
        of the corresponding episode
    :param initial_v: initialization value of all V-values
    :return:
        - V: empirical estimates of the V-values
        - stats: dictionary of statistics collected during training
    """
    # V-values map
    # Dictionary that maps the tuple representation of the state to a dictionary of state values
    V = defaultdict(lambda: initial_v)  # All V-values are initialized to initial_v
    # Stats of training
    episode_rewards = np.empty(num_episodes)
    episode_lengths = np.empty(num_episodes)

    assert action_policy in ("epsilon_greedy", "softmax_")

    if epsilon_exploration_rule is None:
        def epsilon_exploration_rule(n):
            return epsilon_exploration  # if an exploration rule is not given, it is the constant one

    for itr in range(num_episodes):
        env.reset()
        length = 0
        e = defaultdict(lambda: 0)  # shadow variable for each state action pair, all to 0

        # first state and action outside the loop
        state = env.get_state()
        e[state] = 1
        action = eval(action_policy)(env, epsilon_exploration_rule(itr + 1), Q=None)
        reward = 0
        while not env.end:

            # rescale all traces
            for key in list(e):
                e[key] *= trace_decay

            # Move according to the policy
            next_state, reward = env.do_action(action)
            length += 1

            # current state action pair
            e[state] += 1

            if not env.end:
                next_action = eval(action_policy)(env, epsilon_exploration_rule(itr + 1), Q=None)
                target = reward + gamma * V[next_state]
            else:
                target = reward  # the fictitious V-value of V(next_state) is zero

            # update all V-values
            for key in list(e):
                V[key] += alpha * (target - V[state]) * e[key]

            # Preparing for the next move
            state = next_state

            if not env.end:
                action = next_action

        episode_rewards[itr] = reward  # reward of the current episode
        episode_lengths[itr] = length  # length of the current episode

    # Dictionary of stats
    stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }

    return V, stats

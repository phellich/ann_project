import PIL
import io
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from RL_algorithms.algorithms import *


def play(env, Q):
    """
    Utility function to check the correct training of the agent. Shows the path taken by the agent moving in the
    environment env with a greedy policy based on the Q-values Q
    :param env: environment
    :param Q: empirical estimates of the Q-values
    :return:
        - a sequence of snapshots corresponding to the different agent positions
    """
    im_vec = []
    env.reset()
    # Step 0
    fig, _ = env.render(show=False)
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    im = np.asarray(PIL.Image.open(buf))
    im_vec.append(im)
    # Steps 1 to end
    while not env.end:
        action = epsilon_greedy(env, 0, Q)
        env.do_action(action)
        fig, _ = env.render(show=False)
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        im = np.asarray(PIL.Image.open(buf))
        im_vec.append(im)

    fig = plt.figure(figsize=(10, 7))
    columns = 4
    rows = int(np.ceil(len(im_vec)/4))
    for i, img in enumerate(im_vec):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Step " + str(i))
    plt.tight_layout()
    plt.show()


################################# PLOTTING UTILS #################################
def running_average(stats, key):
    """
    Computes the running average of the quantities stored in stats[key]
    :param stats: statistics collected during training
    :param key: key
    :return:
        - the running average (computed after each item) of stats[key]
    """
    return [np.sum(stats[key][:i]) / (i+1) for i in range(len(stats[key]))]


def average_stats(stats_dict, num_avg):
    """
    Computes average quantities (over num_avg training runs) of statistics
    :param stats_dict: dictionary of statistics collected during training e.g. for different random seeds
    :param num_avg: number of averages considered
    :return:
        - mean of the quantity of interest over num_avg training runs
        - standard deviation of the same values
    """
    return np.mean([stats_dict[i] for i in range(num_avg)], axis=0), np.std([stats_dict[i] for i in range(num_avg)], axis=0) / np.sqrt(num_avg)


def compare_episodes_lengths_and_rewards(env, algos, num_avg, show_std, additional_params=None):
    """
    Utility function to compare different algorithms and their performance in terms of average episode length
    :param additional_params: additional parameters common to all algorithms
    :param env: environment
    :param algos: algorithms to be compared
    :param num_avg: number of averages to be performed for different random seeds
    :param show_std: True to show shaded region for the standard deviation in the produced plot
    :return:
    """
    # set figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(14.4, 4.8), squeeze=False)
    # fig.tight_layout(pad=7.)
    fig.subplots_adjust(top=0.9, left=0.1, right=1.1, bottom=0.12)  # adjust the spacing between subplots
    fig.suptitle(f"{env.get_num_states()} states", fontsize=20)
    if additional_params is None:
        additional_params = [{}] * len(algos)
    # loop over the algorithms
    for ind, algo in enumerate(algos):
        name = algo["name"]
        episode_lengths = defaultdict(lambda: {})  # initialize dictionary for current algorithm
        episode_rewards = defaultdict(lambda: {})
        algos_params = {'env': env, **algo["params"], **additional_params[ind]}
        for i in range(num_avg):
            Q, stats = eval(algo["algo_name"])(**algos_params)  # evaluate algorithm
            episode_lengths.update({i: stats["episode_lengths"]})  # save stats of current training
            episode_rewards.update({i: running_average(stats, "episode_rewards")})
        # compute mean and standard deviation
        episodes_averages = average_stats(episode_lengths, num_avg)  # compute averages for current algorithm
        reward_averages = average_stats(episode_rewards, num_avg)  # compute averages for current algorithm
        ax[0, 0].set_xlabel("Episode", fontsize=15)
        ax[0, 0].set_ylabel("Average episode length", fontsize=15)
        if not show_std:
            ax[0, 0].plot(np.arange(len(episodes_averages[0])), episodes_averages[0], label=f"{name}")
        else:
            ax[0, 0].errorbar(np.arange(len(episodes_averages[0])), episodes_averages[0], episodes_averages[1], label=f"{name}")
        ax[0, 1].set_xlabel("Episode", fontsize=15)
        ax[0, 1].set_ylabel("Running average of the reward", fontsize=15)
        ax[0, 1].set_ylim([1, 2])
        ax[0, 1].plot(np.arange(len(episodes_averages[0])), reward_averages[0], label=f"{name}")
        if show_std:
            ax[0, 1].fill_between(np.arange(additional_params[0]["num_episodes"]),
                                  reward_averages[0]-reward_averages[1], reward_averages[0]+reward_averages[1], alpha=0.2)
        # place legend outside plot (below)
        ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(1.1, -0.15), fancybox=True, shadow=True,
                                ncol=5, fontsize='xx-large')  # unique legend for the two plots


def compare_length_first_episode(env, algorithms, additional_params, variable_vals, variable_name, num_avg=500):
    """
    Utility function to compare the length of the first episode (of pure exploration if initial_q = 0) for different
    exploration strategy
    :param env: environment
    :param algorithms: algorithms to be compared
    :param additional_params: additional parameters common to all the algorithms
    :param variable_vals: values of critical variable which is being tested
    :param variable_name: name of the critical variable
    :param num_avg: number of averages considered
    :return:
    """
    # Initialize list for storing episode lengths of each algorithm
    algo_length = []

    # Loop over algorithms
    for algo in algorithms:
        # Print name of the current algorithm
        print(algo['name'])
        # Initialize vector to store the episode lengths
        episode_length = np.zeros((num_avg, len(additional_params)))
        # Loop over different configurations of the additional params
        for j in range(len(additional_params)):
            input_param = {'env': env, **algo['params'], **additional_params[j]}
            for i in range(num_avg):
                returns = eval(algo['algo_name'])(**input_param)
                # Get the training statistics
                stats = returns[-1]
                # Append the length of the current episode
                episode_length[i][j] = stats['episode_lengths'][0]
        # Append the episode lengths for the current algorithm
        algo_length.append(episode_length)

    # Show results
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel(variable_name)
    ax.set_ylabel('Length of the first episode')

    for a, algo in enumerate(algorithms):
        episode_length = algo_length[a]
        it_mean = np.mean(episode_length, axis=0)
        it_stderr = np.std(episode_length, axis=0) / np.sqrt(num_avg)
        ax.plot(variable_vals, it_mean, label=algo['name'])
        ax.fill_between(variable_vals, it_mean + it_stderr, it_mean - it_stderr, color=plt.gca().lines[-1].get_color(),
                        alpha=0.2)

    ax.legend()
    plt.show()
    return


def count_direct_paths(env, stats):
    """
    Utility function to compute the cumulative sum of the number of times the agent has reached directly the highest
    rewarded state from the origin state in the given environment
    :param env: environment
    :param stats: stats collected during training
    :return:
        - the cumulative sum of a 0-1 array where 0/1 corresponds to not having reached/having reached the goal
          in the minimum number of steps
    """
    return np.cumsum([1 if stats["episode_rewards"][i] == env.highest_reward() and
                      stats["episode_lengths"][i] == env.get_direct_path_len()
                      else 0 for i in range(len(stats["episode_rewards"]))])


def compare_rewards_vs_direct_paths(env, algorithms, additional_params, num_episodes=1000, zoom=False,
                                    single_plots=False, frac_episodes_zoom=0.1):
    """
    Compares rewards' number with number of direct paths from origin
    :param env: environment
    :param algorithms: algorithms to be compared
    :param additional_params: additional parameters common to all the algorithms
    :param num_episodes: number of episodes
    :param zoom: True to do a zoom on the first episodes
    :param single_plots: True to show also single plots
    :param frac_episodes_zoom: percentage of the number of episodes to show in zoomed plots
    :return:
    """
    # Comparing number of direct paths and number of rewards with initialization in origin
    algo_timing = []

    for i, algo in enumerate(algorithms):
        # Evaluate the algorithm
        input_param = {'env': env, **algo['params'], **additional_params[i], 'num_episodes': num_episodes}
        returns = eval(algo['algo_name'])(**input_param)
        # Get the stats
        stats = returns[-1]
        episode_lengths = [0, *stats['episode_lengths']]
        # Compute the episode at which each reward is obtained
        reward_time = np.cumsum(episode_lengths)
        # Compute the cumulative reward
        rewards = np.arange(len(reward_time))
        # Compute the number of direct paths
        direct_paths = [item[0] == env.direct_path_len() and item[1] == env.highest_reward() for item in
                        list(zip(episode_lengths, [0, *stats["episode_rewards"]]))]
        direct_paths_sum = np.cumsum(direct_paths)
        # Store the computations
        algo_timing.append({'reward_time': reward_time, 'rewards': rewards, 'direct_paths': direct_paths_sum})

    # Prepare the plots
    fig, ax = plt.subplots(figsize=(8, 5))
    if single_plots:
        fig_paths, ax_paths = plt.subplots(figsize=(8, 5))
    if zoom:
        fig_zoom, ax_zoom = plt.subplots(figsize=(8, 5))
        ax_zoom.set_xlabel('Time (steps)')
        ax2_zoom = ax_zoom.twinx()
        ax_zoom.set_ylabel('Cumulative rewards (-)')
        ax2_zoom.set_ylabel('Number of direct paths from origin (- -)')
        fig_paths_zoom, ax_paths_zoom = plt.subplots(figsize=(8, 5))
        ax_paths_zoom.set_xlabel('Episodes')
        ax_paths_zoom.set_ylabel('Number of direct paths from origin')

    # Show results
    ax.set_xlabel('Time (steps)')
    ax2 = ax.twinx()
    ax.set_ylabel('Number of rewards (-)')
    ax2.set_ylabel('Number of direct paths from origin (- -)')

    idx = int(num_episodes * frac_episodes_zoom)

    if single_plots:
        ax_paths.set_xlabel('Episodes')
        ax_paths.set_ylabel('Number of direct paths from origin')

    idx = int(num_episodes * frac_episodes_zoom)

    for a, algo in enumerate(algorithms):
        reward_time = algo_timing[a]['reward_time']
        rewards = algo_timing[a]['rewards']
        direct_paths = algo_timing[a]['direct_paths']
        color = next(ax._get_lines.prop_cycler)['color']
        
        # Whole training
        # Independent variable: time steps
        ax.plot(reward_time, rewards, '-', label=algo['name'], color=color)
        ax2.plot(reward_time, direct_paths, '--', color=color)
        # Independent variable: episodes
        if single_plots:
            ax_paths.plot(np.arange(len(direct_paths))+1, direct_paths, '-', label=algo['name'], color=color)
        
        # Zoom in
        if zoom:
            ax_zoom.plot(reward_time[:idx], rewards[:idx], '-', label=algo['name'], color=color)
            ax2_zoom.plot(reward_time[:idx], direct_paths[:idx], '--', color=color)
            # Independent variable: episodes
            ax_paths_zoom.plot(np.arange(len(direct_paths[:idx]))+1, direct_paths[:idx], '-', label=algo['name'], color=color)
            ax_paths_zoom.legend()

    ax.legend()
    if single_plots:
        ax_paths.legend()
    if zoom:
        ax_zoom.legend()
        ax_paths_zoom.legend()
    plt.show()

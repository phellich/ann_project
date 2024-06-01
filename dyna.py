from collections import deque
import random
from matplotlib import pyplot as plt
import numpy as np
import os
from pyparsing import col
from tqdm import tqdm
import pandas as pd
import gymnasium as gym

class DynaAgent:
    
    def __init__(self, env, discr_step=[0.025, 0.005], path='dyna_model/', deterministic=False):
        self.path = path
        self.env = env
        self.discr_step = discr_step
        self.deterministic = deterministic
        
        self.positions = np.arange(-1.2, 0.6, self.discr_step[0])
        self.velocities = np.arange(-0.07, 0.07, self.discr_step[1])
        
        self.n_states = len(self.positions) * len(self.velocities)
        self.n_actions = self.env.action_space.n
        
        self.gamma = 0.99
        self.epsilon = 0.9 # exponential decay until 0.05
        self.k = 3
        
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))
        self.Q = np.zeros((self.n_states, self.n_actions))

        self.P_count = np.zeros((self.n_states, self.n_actions))
        
        self.t = 0
        self.replay_buffer = deque(maxlen=10000)

    def save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        np.save(self.path + 'P.npy', self.P)
        np.save(self.path + 'R.npy', self.R)
        np.save(self.path + 'Q.npy', self.Q)

    def load(self):
        self.P = np.load(self.path + 'P.npy')
        self.R = np.load(self.path + 'R.npy')
        self.Q = np.load(self.path + 'Q.npy')
        
    def discretize(self, state):
        position, velocity = state
        position = np.abs(self.positions - position).argmin()
        velocity = np.abs(self.velocities - velocity).argmin()
        return position * len(self.velocities) + velocity
    
    def epsilon_greedy(self, state):
        if self.epsilon > 0.05:
            self.epsilon = 0.9 * np.exp(-0.0018 * self.t)
        self.t += 1 
        if np.random.rand() < self.epsilon and not self.deterministic:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.Q[state])
        
    def select_action(self, state):
        state = self.discretize(state)
        return self.epsilon_greedy(state)
    
    def observe(self, state, action, next_state, reward):
        state = self.discretize(state)
        next_state = self.discretize(next_state)
        self.replay_buffer.append((state, action, next_state, reward))
        
        # Update the model
        self.P[state][action][next_state] += 1
        self.R[state][action] += reward  
        self.P_count[state][action] += 1
        
    def update(self):
        last_state, last_action, _, _ = self.replay_buffer[-1]
        prev_q = self.Q[last_state, last_action]
        new_q = self.update_single(*self.replay_buffer[-1])
        if len(self.replay_buffer) > self.k:
            for _ in range(self.k):
                self.update_single(*random.choice(self.replay_buffer))
        return new_q - prev_q
                
    def update_single(self, state, action, next_state, reward):
        transition_prob = self.P[state][action] / self.P_count[state][action]
        expected_reward = self.R[state][action] / self.P_count[state][action]
        self.Q[state, action] = expected_reward + self.gamma * np.sum(transition_prob * np.max(self.Q, axis=-1))
        return self.Q[state, action]
        
        
    def planning(self):
        for _ in range(self.k):
            state, action, next_state, reward = random.choice(self.replay_buffer)
            self.update_single(state, action, next_state, reward)


def test(env, agent: DynaAgent, ax, color='red'):
    positions = []
    velocities = []
    rewards = []
    done = False
    state, _ = env.reset()
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        position, velocity = state
        positions.append(position)
        velocities.append(velocity)
        rewards.append(reward)

    # heights = np.sin(3 * np.array(positions))
    ax.plot(positions, velocities, label='Position vs Velocity', color=color)
    # plt.show()
    



def train(env, agent: DynaAgent, n_episodes: int=3000, log_interval: int=300):
    episode_durations = []
    episode_rewards = []
    episode_q_steps = []
    for n in tqdm(range(n_episodes)):
        done = False
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_q_step = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.observe(state, action, next_state, reward)
            
            loss = agent.update()
            state = next_state
            episode_length += 1
            episode_q_step += loss
            episode_reward += reward
            done = terminated or truncated
            
        episode_durations.append(episode_length)
        episode_rewards.append(episode_reward)
        episode_q_steps.append(episode_q_step)
        
        if n % log_interval == (log_interval - 1):
            print(f'Episode {n} - Reward: {episode_reward} - Qvalue step: {episode_q_step} - Duration: {episode_length}')

    # Save the model and results  
    agent.save()
    df = pd.DataFrame({'episode': list(range(n+1)), 'reward': episode_rewards, 'duration': episode_durations, 'qstep': episode_q_steps})
    df.to_csv(agent.path + 'episode_results.csv', index=False)


def plot_qvalues(model_path, dyna, ax=None):
    qvalues_path = os.path.join(model_path, 'Q.npy')
    qvalues = np.load(qvalues_path)
    qvalues[qvalues == 0] = np.nan

    n_positions = len(dyna.positions)
    n_velocities = len(dyna.velocities)

    X, Y = np.meshgrid(dyna.positions, dyna.velocities, indexing='ij')
    Z = qvalues.max(axis=-1).reshape(n_positions, n_velocities)

# Scatter plot of Q-values with colorbar
    _, ax = plt.subplots(figsize=(10, 10 * n_velocities / n_positions)) if ax is None else (None, ax)
    sc = ax.scatter(X, Y, c=Z, s=1000 / np.sqrt(n_positions))
    plt.colorbar(sc)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    plt.title('Learned Q-values')
    return ax

    # Heatmap of Q-values
    # fig, ax = plt.subplots(figsize=(10, 6))
    # plt.imshow(Z, extent=(min(dyna.positions), max(dyna.positions), 
                      # min(dyna.velocities), max(dyna.velocities)), 
                    # aspect='auto', origin='lower', interpolation='bicubic')
    # plt.colorbar()
    # ax.set_xlabel('Position')
    # ax.set_ylabel('Velocity')
    # plt.show()


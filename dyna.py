from collections import deque
import random
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import gymnasium as gym

class DynaAgent:
    
    def __init__(self, env, discr_step=[0.025, 0.005], path='dyna_model/'):
        self.path = path
        self.env = env
        self.discr_step = discr_step
        
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
        if np.random.rand() < self.epsilon:
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
        self.R[state][action] = reward
        
    def update(self):
        last_state, last_action, _, _ = self.replay_buffer[-1]
        prev_q = self.Q[last_state, last_action]
        new_q = self.update_single(*self.replay_buffer[-1])
        if len(self.replay_buffer) > self.k:
            for _ in range(self.k):
                self.update_single(*random.choice(self.replay_buffer))
        return new_q - prev_q
                
    
    def update_single(self, state, action, next_state, reward):
        transition = self.P[state][action] / np.sum(self.P[state][action])
        self.Q[state, action] = self.R[state][action] + self.gamma * np.sum(transition * np.max(self.Q, axis=-1))
        return self.Q[state, action]
        
        
    def planning(self):
        for _ in range(self.k):
            state, action, next_state, reward = random.choice(self.replay_buffer)
            self.update_single(state, action, next_state, reward)



def train(env, agent: DynaAgent, n_episodes: int=3000):
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
        
        if n % 300 == 0 or n == n_episodes - 1:
            print(f'Episode {n} - Reward: {episode_reward} - Qvalue step: {episode_q_step} - Duration: {episode_length}')
            agent.save()
            
            # Save the results
            df = pd.DataFrame({'episode': list(range(n+1)), 'reward': episode_rewards, 'duration': episode_durations, 'qstep': episode_q_steps})
            df.to_csv(agent.path + 'episode_results.csv', index=False)

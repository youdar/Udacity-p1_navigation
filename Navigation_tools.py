from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import torch
import sys
import os

import matplotlib.pyplot as plt

from dqn_agent import Agent

LOCAL_PATH = '/Users/youval.dar/Documents/workspace/deep-reinforcement-learning/Udacity-p1_navigation'

"""
Use this code to be able to better utilize debugging tools
Then copy to Navigation.ipynb
"""


class BananaGame():

    def __init__(self):
        self.env = None
        self.brain = None
        self.brain_name = None
        self.agent = None

    def start_env(self):
        self.env = UnityEnvironment(file_name=os.path.join(LOCAL_PATH, "Banana.app"), worker_id=2, seed=1)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        return self.env

    @staticmethod
    def print_scores_stats(scores):
        """
        Prints basic states of data in the list "scores"

        Args:
            scores (list): list of scores
        """
        scores = np.array(scores)
        print('Uniformly random action')
        print('-----------------------')
        print('Avg score: {:.2f}'.format(np.average(scores)))
        print('Min score:', np.min(scores))
        print('Max score:', np.max(scores))
        print('Median score:', np.median(scores))

    def uniform_random(self, n_episodes=10):
        env = self.start_env()
        print('Brain name:', self.brain_name)
        env_info = env.reset(train_mode=True)[self.brain_name]

        # number of agents in the environment
        print('Number of agents:', len(env_info.agents))
        # number of actions
        action_size = self.brain.vector_action_space_size
        print('Number of actions:', action_size)
        # examine the state space
        state = env_info.vector_observations[0]
        print('States look like:', state)
        state_size = len(state)
        print('States have length:', state_size)

        scores = []
        for i in range(n_episodes):
            print('Game:', i + 1)
            env_info = env.reset(train_mode=False)[self.brain_name]  # reset the environment
            state = env_info.vector_observations[0]  # get the current state
            score = 0  # initialize the score
            done = False  # episode finished
            while not done:
                action = np.random.randint(action_size)  # select an action
                env_info = env.step(action)[self.brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                score += reward  # update the score
                state = next_state  # roll over the state to next time step

            print("{}: Score: {}".format(i + 1, score))
            sys.stdout.flush()
            scores.append(score)

        # Print games summary
        self.print_scores_stats(scores)
        self.env.close()

    def training_model(self, model_num=1):
        # Using DQN from the file dqn_agent.py
        env = self.start_env()

        # reset the environment
        env_info = env.reset(train_mode=True)[self.brain_name]
        state = env_info.vector_observations[0]
        state_size = len(state)
        action_size = self.brain.vector_action_space_size

        # Set agent params
        agent = Agent(
            state_size=state_size,
            action_size=action_size,
            seed=0,
            model_num=model_num)

        scores = self.dqn(agent, model_num=model_num)
        self.env.close()
        return scores

    def dqn(self, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, model_num=1):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start  # initialize epsilon
        for i_episode in range(1, n_episodes + 1):
            env_info = self.env.reset(train_mode=False)[self.brain_name]
            state = env_info.vector_observations[0]
            score = 0
            for t in range(max_t):
                action = agent.act(state, eps)
                env_info = self.env.step(action)[self.brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(eps_end, eps_decay * eps)  # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= 15.5:
                str_out = '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
                print(str_out.format(i_episode - 100, np.mean(scores_window)))

                if model_num == 1:
                    fn = 'checkpoint.pth'
                else:
                    fn = 'checkpoint_dropout.pth'
                fn = os.path.join(LOCAL_PATH, fn)
                torch.save(agent.qnetwork_local.state_dict(), fn)
                break
        return scores

    def trained_model(self, n_episodes=10, model_num=1):
        # reset the environment
        env = self.start_env()
        env_info = env.reset(train_mode=False)[self.brain_name]
        state = env_info.vector_observations[0]
        state_size = len(state)
        action_size = self.brain.vector_action_space_size

        # Set agent params
        agent = Agent(
            state_size=state_size,
            action_size=action_size,
            seed=0,
            model_num=model_num)

        # load the weights from file
        if model_num == 1:
            fn = 'checkpoint.pth'
        else:
            fn = 'checkpoint_dropout.pth'
        fn = os.path.join(LOCAL_PATH, fn)
        agent.qnetwork_local.load_state_dict(torch.load(fn))

        scores = []
        for i in range(n_episodes):
            env_info = env.reset(train_mode=False)[self.brain_name]  # reset the environment
            state = env_info.vector_observations[0]  # get the current state
            score = 0  # initialize the score
            done = False
            while not done:
                action = agent.act(state)
                env_info = env.step(action)[self.brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                score += reward  # update the score
                state = next_state  # roll over the state to next time step

            scores.append(score)
            print("{}: Score: {}, Avg score: {}".format(i + 1, score, round(sum(scores) / len(scores)), 1))
            sys.stdout.flush()

        # Print games summary
        self.print_scores_stats(scores)
        self.env.close()
        return scores

    def plot_scores(self, scores):
        """
        plot the scores

        Args:
            scores (list): List of scores
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()


if __name__ == '__main__':
    o = BananaGame()
    # o.uniform_random(3)
    # scores = o.training_model(model_num=2)
    # o.plot_scores(scores)
    scores = o.trained_model(10, model_num=2)
    # o.plot_scores(scores)

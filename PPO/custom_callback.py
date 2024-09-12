import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import pickle
import tensorflow as tf


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.states = []
        self.actions = []
        self.rewards = []

    def _on_step(self) -> bool:
        obs = self.locals['obs_tensor']
        actions = self.locals['actions']
        rewards = self.locals['rewards']
        self.states.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)

        return True

    def on_training_end(self):
        dataset = {}
        if 'rewards' not in dataset:
            dataset['rewards'] = []
        if 'actions' not in dataset:
            dataset['actions'] = []
        if 'states' not in dataset:
            dataset['states'] = []
        dataset['states'] = self.states
        dataset['actions'] = self.actions
        dataset['rewards'] = self.rewards
        with open('data.pkl', 'wb') as file:
          pickle.dump(dataset, file)


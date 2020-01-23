import gym as gym
import numpy as np

from drl.environments.util import to_bin


class DiscretizedCartPole:
    def __init__(self, bins=9):
        self.env = gym.make('CartPole-v0')
        self.done = False
        self.bins = bins
        self.cart_position_bins = np.linspace(-2.4, 2.4, self.bins)
        self.cart_velocity_bins = np.linspace(-2, 2, self.bins)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, self.bins)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, self.bins)
        self.starting_state = self.transform(self.reset())

    def reset(self):
        self.starting_state = self.transform(self.env.reset())
        self.done = False
        return self.starting_state

    def transform(self, observation):
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        return [
            to_bin(cart_pos, self.cart_position_bins),
            to_bin(cart_vel, self.cart_velocity_bins),
            to_bin(pole_angle, self.pole_angle_bins),
            to_bin(pole_vel, self.pole_velocity_bins)
        ]

    def is_terminal(self):
        return self.done

    def generate_actions(self):
        for action in range(self.env.action_space.n):
            yield action

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation_binned = self.transform(observation)
        self.done = done
        return observation_binned, reward, done, info


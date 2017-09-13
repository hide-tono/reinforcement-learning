# coding=utf-8
import gym
import numpy
from gym import spaces


class SimpleGraph(gym.Env):
    metadata = {'render.modes': ['human', 'array']}

    def __init__(self):
        self.state_list = ['s1', 's2', 's3', 's4']
        self.action_list = ['a1', 'a2']
        self.state = 0
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=3, shape=numpy.shape([4]))
        self.reword_list = numpy.array([
            [0, 1],
            [-1, 1],
            [5, -100],
            [0, 0]
        ])
        self.next_state_list = numpy.array([
            [2, 1],
            [0, 3],
            [3, 0],
            [3, 3]
        ])

    def _reset(self):
        self.state = 0
        return self.make_obs()

    def _step(self, action):
        if self.state_list[self.state] == 's4':
            return [0, 0, 0, 1], 0, True, None
        reword = self.reword_list[self.state][action]
        self.state = self.next_state_list[self.state][action]

        return self.make_obs(), reword, self.state_list[self.state] == 's4', None

    def _render(self, mode='human', close=False):
        return self.make_obs()

    def make_obs(self):
        obs = numpy.zeros(4)
        obs[self.state] = 1

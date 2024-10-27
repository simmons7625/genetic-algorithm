from collections import deque
import random

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, observation, action, reward, next_observation):
        experience = (observation, action, reward, next_observation)
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        observations, actions, rewards, next_observations = zip(*experiences)
        return observations, actions, rewards, next_observations
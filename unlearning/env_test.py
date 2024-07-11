import gym
import d4rl # Import required to register environments, you may need to also import the submodule
import numpy as np
import d4rl.gym_mujoco

# Create the environment
env = gym.make('halfcheetah-medium-expert-v0')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()

print(dataset['observations'].shape[0]) # An N x dim_observation Numpy array of observations

env = gym.make('hopper-medium-expert-v0')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()

print(dataset['observations'].shape[0]) # An N x dim_observation Numpy array of observations

env = gym.make('walker2d-medium-expert-v0')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()

print(dataset['observations'].shape[0]) # An N x dim_observation Numpy array of observations
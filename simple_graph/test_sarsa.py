import gym
import random

from keras.optimizers import Adam
from rl.agents.sarsa import SarsaAgent

from keras.layers import Dense, Activation
from keras.models import Sequential
from rl.policy import BoltzmannQPolicy

env = gym.make('SimpleGraph-v0')

Episodes = 200

obs = []
input_shape = 4
nb_actions = 2
model = Sequential()
model.add(Dense(input_shape=[input_shape], units=2))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))


# SARSA does not require a memory.
policy = BoltzmannQPolicy()
sarsa = SarsaAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
sarsa.fit(env, nb_steps=5000, visualize=False, verbose=2)

# After training is done, we save the final weights.
sarsa.save_weights('sarsa_{}_weights.h5f'.format('SimpleGraph-v0'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
sarsa.test(env, nb_episodes=5, visualize=False)
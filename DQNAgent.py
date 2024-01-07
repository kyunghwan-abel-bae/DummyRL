from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.models import Sequential

from collections import deque
import random

import numpy as np

REPLAY_MEMORY_SIZE = 100

class DQNAgent:
    def __init__(self, actions):
        self.ACTIONS = np.array(actions)

        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.gamma = 0.99
        self.batch_size = 5

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.model = self._create_model()

    def _create_model(self):
        model = Sequential([
            # Dense(24, input_dim=1, activation='relu'),
            Dense(24, input_dim=2, activation='relu'),
            Dense(12, activation='relu'),
            Dense(len(self.ACTIONS))
        ])

        model.compile(optimizer='rmsprop', loss='mse')
        return model

    def update_replay_memory(self, current_state, action, reward, next_state, done):
        self.replay_memory.append((current_state, action, reward, next_state, done))

    def ensure_2d(self, array):
        array = np.asarray(array)
        if array.ndim == 1:
            array = np.expand_dims(array, axis=0)
        return array

    def get_q_values(self, state):
        state = self.ensure_2d(state)
        return self.model.predict(state)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.ACTIONS)
        else:
            state = self.ensure_2d(state)
            action = np.argmax(self.get_q_values(state)) # 0, 1, 2, 3 => 1, 2, 3, 4
        return action

    def learn(self, state, action, reward, next_state, next_action):
        if len(self.replay_memory) < 10:
            return

        print("learn")

        samples = random.sample(self.replay_memory, self.batch_size)
        current_input = np.stack([sample[0] for sample in samples])
        print("current_input : ", current_input)
        current_q_values = self.model.predict(current_input)
        next_input = np.stack([sample[3] for sample in samples])
        # next_q_values = self.target_model.predict(next_input) # weights transferred periodically
        next_q_values = self.model.predict(next_input)

        # update q values
        for i, (current_state, action, reward, _, done) in enumerate(samples):
            if done:
                next_q_value = reward
            else:
                next_q_value = reward + self.gamma * np.max(next_q_values[i])
            current_q_values[i, action] = next_q_value

        # fit model
        hist = self.model.fit(current_input, current_q_values, batch_size=self.batch_size, verbose=0, shuffle=False)




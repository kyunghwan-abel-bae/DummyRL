import torch
from torch import nn

from einops import rearrange

from snake import NUM_CHANNELS, NUM_ACTIONS
from collections import deque
import random
import numpy as np


class DQNAgentModel(nn.Module):
    def __init__(self, field_width, field_height, num_channels, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (field_height - 4) * (field_width - 4), 256)
        self.fc2 = nn.Linear(256, num_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)

        print(f"x.shape : {x.shape}")
        x = rearrange(x, 'b h w c -> b c h w')

        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x#.detach().numpy()


class DQNAgent:
    def __init__(self, field_size, gamma, batch_size, min_replay_memory_size, replay_memory_size, target_update_freq):
        self.gamma = gamma
        self.field_height, self.field_width = field_size
        self.batch_size = batch_size
        self.min_replay_memory_size = min_replay_memory_size
        self.target_update_freq = target_update_freq

        self.model = DQNAgentModel(self.field_width, self.field_height, NUM_CHANNELS, NUM_ACTIONS)
        self.target_model = DQNAgentModel(self.field_width, self.field_height, NUM_CHANNELS, NUM_ACTIONS)
        self.target_model.load_state_dict(self.model.state_dict())

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters())
        # self.target_model.set_weights(self.model.get_weights())
        # self.model.summary()

        self.replay_memory = deque(maxlen=replay_memory_size)
        self.target_update_counter = 0

        # quit()

    # def _create_model(self):
    #     model = Sequential([
    #         Conv2D(32, (3, 3), input_shape=(self.field_height, self.field_width, NUM_CHANNELS), activation='relu'),
    #         Dropout(0.1),
    #         Conv2D(32, (3, 3), activation='relu'),
    #         Dropout(0.1),
    #         Flatten(),
    #         Dense(256, activation='relu'),
    #         Dropout(0.1),
    #         Dense(NUM_ACTIONS)
    #     ])
    #     model.compile(optimizer='rmsprop', loss='mse')
    #     return model

    def update_replay_memory(self, current_state, action, reward, next_state, done):
        self.replay_memory.append((current_state, action, reward, next_state, done))

    def get_q_values(self, x):
        # print(f"x : {x.shape}, state : {state.shape}")
        # quit()
        q_values = self.model(x)
        return q_values.detach().numpy()
        # return self.model(x)

    def train(self):
        # guarantee the minimum number of samples
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        # get current q values and next q values
        samples = random.sample(self.replay_memory, self.batch_size)
        current_input = np.stack([sample[0] for sample in samples])
        current_q_values = self.model(current_input)
        next_input = np.stack([sample[3] for sample in samples])
        # next_q_values = self.target_model.predict(next_input)
        next_q_values = self.model(next_input)

        # update q values
        for i, (current_state, action, reward, _, done) in enumerate(samples):
            if done:
                next_q_value = reward
            else:
                next_q_value = reward + self.gamma * torch.max(next_q_values[i])
            current_q_values[i, action] = next_q_value

        # fit model
        # hist = self.model.fit(current_input, current_q_values, batch_size=self.batch_size, verbose=0, shuffle=False)
        # pred_tensor = torch.tensor(next_q_values, dtype=torch.float32)
        # target_tensor = torch.tensor(current_q_values, dtype=torch.float32)
        pred_tensor = next_q_values
        target_tensor = current_q_values

        # print(f"pred_tensor.shape : {pred_tensor.shape}, target_tensor.shape : {target_tensor.shape}")

        loss = self.loss_fn(pred_tensor, target_tensor)

        # print(f"loss : {loss}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # print(f"loss backward : {loss}")
        # loss = hist.history['loss'][0]
        return loss

    '''
    # def increase_target_update_counter(self):
    #     self.target_update_counter += 1
    #     if self.target_update_counter >= self.target_update_freq:
    #         self.target_model.set_weights(self.model.get_weights())
    #         self.target_update_counter = 0

    # def save(self, model_filepath, target_model_filepath):
    #     self.model.save(model_filepath)
    #     self.target_model.save(target_model_filepath)

    # def load(self, model_filepath, target_model_filepath):
    #     self.model = keras.models.load_model(model_filepath)
    #     self.target_model = keras.models.load_model(target_model_filepath)
'''
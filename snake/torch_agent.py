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

        x = rearrange(x, 'b h w c -> b c h w')

        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


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

        self.replay_memory = deque(maxlen=replay_memory_size)
        self.target_update_counter = 0

    def update_replay_memory(self, current_state, action, reward, next_state, done):
        self.replay_memory.append((current_state, action, reward, next_state, done))

    def get_q_values(self, x):
        q_values = self.model(x)
        return q_values.detach().numpy()

    def train(self):
        # guarantee the minimum number of samples
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        # get current q values and next q values
        samples = random.sample(self.replay_memory, self.batch_size)
        current_input = np.stack([sample[0] for sample in samples])
        current_q_values = self.model(current_input)
        current_q_backup_values = current_q_values.clone()
        next_input = np.stack([sample[3] for sample in samples])
        next_q_values = self.target_model(next_input)

        # update q values
        for i, (current_state, action, reward, _, done) in enumerate(samples):
            if done:
                next_q_value = reward
            else:
                next_q_value = reward + self.gamma * torch.max(next_q_values[i])
            current_q_values[i, action] = next_q_value

        pred_tensor = current_q_backup_values
        target_tensor = current_q_values

        loss = self.loss_fn(pred_tensor, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def increase_target_update_counter(self):
        self.target_update_counter += 1
        if self.target_update_counter >= self.target_update_freq:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

    def load(self, model_filepath):
        self.model.load_state_dict(torch.load(model_filepath))
        self.target_model.load_state_dict(torch.load(model_filepath))
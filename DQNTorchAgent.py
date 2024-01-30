import torch
from torch import nn

from collections import deque
import random

import numpy as np

REPLAY_MEMORY_SIZE = 100

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device.")


class DQNTorchModel(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 12),
            nn.ReLU(),
            nn.Linear(12, 12),
            nn.ReLU(),
            nn.Linear(12, num_actions)
        )

    def forward(self, X):
        return self.linear_relu_stack(X)


class DQNTorchAgent:
    def __init__(self, actions):

        self.ACTIONS = np.array(actions)

        self.epsilon = 0.1 # By using epsilon decaying, learning can be stablized.
        self.gamma = 0.99
        self.batch_size = 5

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.model = DQNTorchModel(len(actions)).to(device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters())

    def update_replay_memory(self, current_state, action, reward, next_state, done):
        self.replay_memory.append((current_state, action, reward, next_state, done))

    def decay_epsilon(self, rate):
        self.epsilon *= rate

    def get_q_values(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        q_values = self.model(state)
        return q_values.detach().cpu().numpy()

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.ACTIONS)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = self.model(state)
            action = torch.argmax(q_values).item() # 0, 1, 2, 3 => 1, 2, 3, 4

        return action

    def learn(self):
        if len(self.replay_memory) < 10:
            return

        samples = random.sample(self.replay_memory, self.batch_size)

        current_input = torch.tensor([sample[0] for sample in samples], dtype=torch.float32).to(device)
        current_q_values = self.model(current_input)

        next_input = torch.tensor([sample[3] for sample in samples], dtype=torch.float32).to(device)
        next_q_values = self.model(next_input)

        for i, (current_state, action, reward, _, done) in enumerate(samples):
            if done:
                next_q_value = reward
            else:
                next_q_value = reward + self.gamma * torch.max(next_q_values[i])
            current_q_values[i, action] = next_q_value

        pred_tensor = next_q_values
        target_tensor = current_q_values
        loss = self.loss_fn(pred_tensor, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

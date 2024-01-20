import torch
from torch import nn
from torchvision.transforms import ToTensor

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

        self.flatten = nn.Flatten()
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

        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.gamma = 0.99
        self.batch_size = 5

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.model = DQNTorchModel(len(actions))
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=1e-3)

    def update_replay_memory(self, current_state, action, reward, next_state, done):
        self.replay_memory.append((current_state, action, reward, next_state, done))

    def ensure_2d(self, array):
        array = np.asarray(array)
        if array.ndim == 1:
            array = np.expand_dims(array, axis=0)

        tensor = torch.from_numpy(array).float()  # numpy 배열을 torch.Tensor로 변환하고 데이터 타입을 float로 설정
        return tensor

    def get_q_values(self, state):
        state = self.ensure_2d(state)
        q_values = self.model(state)
        return q_values.detach().numpy()

    def get_action(self, state):
        # print("get_action")
        # print(f"np.random.rand() : {np.random.rand()}, self.epsilon : {self.epsilon}")
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.ACTIONS)
        else:
            state = self.ensure_2d(state)
            action = np.argmax(self.get_q_values(state)) # 0, 1, 2, 3 => 1, 2, 3, 4
        # print(f"action : {action}")
        return action

    def learn(self, state, action, reward, next_state, next_action):
        if len(self.replay_memory) < 10:
            return

        print("learn")

        samples = random.sample(self.replay_memory, self.batch_size)
        current_input = np.stack([sample[0] for sample in samples])
        print("current_input : ", current_input)

        tensor_input = self.ensure_2d(current_input)
        current_q_values = self.model(tensor_input)
        current_q_values = current_q_values.detach().numpy()

        next_input = np.stack([sample[3] for sample in samples])
        tensor_next_input = self.ensure_2d(next_input)
        next_q_values = self.model(tensor_next_input)
        next_q_values = next_q_values.detach().numpy()

        # update q values
        for i, (current_state, action, reward, _, done) in enumerate(samples):
            if done:
                next_q_value = reward
            else:
                next_q_value = reward + self.gamma * np.max(next_q_values[i])
            current_q_values[i, action] = next_q_value

        pred_tensor = torch.from_numpy(next_q_values).float()  # numpy 배열을 torch.Tensor로 변환하고 데이터 타입을 float로 설정
        target_tensor = torch.from_numpy(current_q_values).float()  # numpy 배열을 torch.Tensor로 변환하고 데이터 타입을 float로 설정
        # pred = next_q_values
        # target = current_q_values
        loss = self.loss_fn(pred_tensor, target_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()







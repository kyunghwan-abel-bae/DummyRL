import torch
import torch.nn as nn
import torch.optim as optim
import random

REPLAY_MEMORY_SIZE = 100

class DQNTorchAgent2(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNTorchAgent2, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 12)
        self.fc3 = nn.Linear(12, output_size)

        self.optimizer = optim.RMSprop(self.parameters())
        self.loss_function = nn.MSELoss()

        self.replay_memory = []

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def update_replay_memory(self, current_state, action, reward, next_state, done):
        self.replay_memory.append((current_state, action, reward, next_state, done))
        if len(self.replay_memory) > REPLAY_MEMORY_SIZE:
            self.replay_memory.pop(0)

    def get_q_values(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        return self(state).detach().numpy()

    def get_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(range(self.fc3.out_features))
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self(state)
            return torch.argmax(q_values).item()

    def learn(self, gamma=0.99, batch_size=5):
        if len(self.replay_memory) < batch_size:
            return

        samples = random.sample(self.replay_memory, batch_size)
        current_input = torch.tensor([sample[0] for sample in samples], dtype=torch.float32)
        current_q_values = self(current_input)

        next_input = torch.tensor([sample[3] for sample in samples], dtype=torch.float32)
        next_q_values = self(next_input).detach()

        for i, (current_state, action, reward, _, done) in enumerate(samples):
            if done:
                next_q_value = reward
            else:
                next_q_value = reward + gamma * torch.max(next_q_values[i])
            current_q_values[i][action] = next_q_value

        self.optimizer.zero_grad()
        loss = self.loss_function(current_q_values, current_q_values.detach())
        loss.backward()
        self.optimizer.step()

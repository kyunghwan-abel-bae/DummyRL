from Python_RL_Envs.GridWorld import *
from DQNTorchAgent import DQNTorchAgent
from DQNTorchAgent2 import DQNTorchAgent2
from matplotlib import pyplot as plt

env = GridWorld(5, 5, state_mode="absolute", goal_included_in_state=False)
env.add_obstacles(1, 1, 0, include_state=False)
env.add_obstacles(0, 3, 0, include_state=False)
env.add_obstacles(3, 0, 0, include_state=False)
env.add_obstacles(4, 2, 0, include_state=False)
env.add_obstacles(2, 4, 0, include_state=False)

agent = DQNTorchAgent2(2, 4)

episodes = []
scores = []

for E in range(1000):
# for E in range(10):
# for E in range(1000):
    state = env.reset()
    done = False

    score = 0

    while not done:
        action = agent.get_action(state)

        next_state, reward, done = env.step((action+1), show=False)
        # print("update_replay_memory : ", state)
        agent.update_replay_memory(state, action, reward, next_state, done)

        next_action = agent.get_action(next_state)

        # For DQNAgent1
        # agent.learn(state, action, reward, next_state, next_action)

        # For DQNAgent2
        agent.learn()

        score += reward

        state = next_state
        # break

    print(f"episode {E} - score {score}")
    scores.append(score)
    episodes.append(E)

plt.plot(episodes, scores)
plt.show()
# '''
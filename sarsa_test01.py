from Python_RL_Envs.GridWorld import *
from SARSAAgent import SARSA
from matplotlib import pyplot as plt

env = GridWorld(5, 5, state_mode="absolute", goal_included_in_state=False)
env.add_obstacles(1, 1, 0, include_state=False)
env.add_obstacles(0, 3, 0, include_state=False)
env.add_obstacles(3, 0, 0, include_state=False)
env.add_obstacles(4, 2, 0, include_state=False)
env.add_obstacles(2, 4, 0, include_state=False)

#env.show([0, 0], 0, 0)

agent = SARSA([1, 2, 3, 4])

episodes = []
scores = []

for E in range(1000):
    state = env.reset()
    done = False

    score = 0

    while not done:
        action = agent.get_action(state)

        next_state, reward, done = env.step(action, show=True)
        next_action = agent.get_action(next_state)

        agent.learn(state, action, reward, next_state, next_action)

        score += reward

        state = next_state

    print(f"episode {E} - score {score}")
    scores.append(score)
    episodes.append(E)

plt.plot(episodes, scores)
plt.show()
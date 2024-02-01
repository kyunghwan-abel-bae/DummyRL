import time

from Python_RL_Envs.GridWorld import *
from DQNTorchAgent import DQNTorchAgent
from matplotlib import pyplot as plt

env = GridWorld(5, 5, state_mode="absolute", goal_included_in_state=False)
env.add_obstacles(1, 1, 0, include_state=False)
env.add_obstacles(0, 3, 0, include_state=False)
env.add_obstacles(3, 0, 0, include_state=False)
env.add_obstacles(4, 2, 0, include_state=False)
env.add_obstacles(2, 4, 0, include_state=False)

agent = DQNTorchAgent([0,1,2,3])

episodes = []
scores = []
start_time = time.time()
for E in range(1000):
    state = env.reset()
    done = False

    score = 0

    if E % 100 == 0:
        agent.decay_epsilon(0.5)

    while not done:
        action = agent.get_action(state)

        next_state, reward, done = env.step((action+1), show=False)

        agent.update_replay_memory(state, action, reward, next_state, done)

        next_action = agent.get_action(next_state)

        agent.learn()

        score += reward

        state = next_state

    print(f"episode {E} - score {score}")
    scores.append(score)
    episodes.append(E)

elapsed_time = time.time()-start_time
print(f"Elapsed Time(seconds) : {elapsed_time}")

agent.save_model("torch_test")

plt.plot(episodes, scores)
plt.show()
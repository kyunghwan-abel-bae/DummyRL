import random
import numpy as np
import tensorflow as tf
from torch_agent import DQNAgent
from tqdm import tqdm
from snake import Snake, NUM_ACTIONS
import pickle
import os
from summary import Summary
from level_loader import LevelLoader
from matplotlib import pyplot as plt
import time
import torch

class DQNTrainer:
    def __init__(self,
                 level_filepath,
                 episodes=30000,
                 initial_epsilon=1.,
                 min_epsilon=0.1,
                 exploration_ratio=0.5,
                 max_steps=2000,
                 render_freq=500,
                 enable_render=True,
                 render_fps=20,
                 save_dir='checkpoints',
                 enable_save=True,
                 save_freq=500,
                 gamma=0.99,
                 batch_size=64,
                 min_replay_memory_size=1000,
                 replay_memory_size=100000,
                 target_update_freq=5,
                 seed=42
                 ):
        self.set_random_seed(seed)

        self.episodes = episodes
        self.max_steps = max_steps
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.exploration_ratio = exploration_ratio
        self.render_freq = render_freq
        self.enable_render = enable_render
        self.render_fps = render_fps
        self.save_dir = save_dir
        self.enable_save = enable_save
        self.save_freq = save_freq

        if enable_save and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        level_loader = LevelLoader(level_filepath)

        self.agent = DQNAgent(
            level_loader.get_field_size(),
            gamma=gamma,
            batch_size=batch_size,
            min_replay_memory_size=min_replay_memory_size,
            replay_memory_size=replay_memory_size,
            target_update_freq=target_update_freq
        )
        self.env = Snake(level_loader)
        self.summary = Summary()
        self.current_episode = 0
        self.max_average_length = 0

        self.epsilon_decay = (initial_epsilon-min_epsilon)/(exploration_ratio*episodes)

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)

    def train(self):
        pbar = tqdm(initial=self.current_episode, total=self.episodes, unit='episodes')

        list_episodes = []
        list_loss = []
        list_reward = []

        while self.current_episode < self.episodes:
            current_state = self.env.reset()

            done = False
            steps = 0
            total_loss = 0.0

            while not done and steps < self.max_steps:
                if random.random() > self.epsilon:
                    action = np.argmax(self.agent.get_q_values(np.array([current_state])))
                else:
                    action = np.random.randint(NUM_ACTIONS)

                next_state, reward, done = self.env.step(action)

                self.agent.update_replay_memory(current_state, action, reward, next_state, done)

                temp_loss = self.agent.train()
                if temp_loss is not None:
                    temp_loss = temp_loss.item()
                    total_loss = total_loss + temp_loss
                self.summary.add('loss', temp_loss)

                current_state = next_state
                steps += 1

            self.agent.increase_target_update_counter()

            self.summary.add('length', self.env.get_length())
            self.summary.add('reward', self.env.tot_reward)
            self.summary.add('steps', steps)

            # decay epsilon
            self.epsilon = max(self.epsilon-self.epsilon_decay, self.min_epsilon)

            self.current_episode += 1

            list_episodes.append(self.current_episode)
            list_loss.append(total_loss)
            list_reward.append(self.env.tot_reward)

            # added by KH - For analysis, trainer is done in middle of training
            if self.current_episode % 20000 == 0:
                # list_loss shows strange values(significant high value)
                # plt.plot(list_episodes, list_loss, label="total loss")
                plt.plot(list_episodes, list_reward, label="reward")
                plt.legend()
                plt.show()
                self.preview(self.render_fps)
                time.sleep(10)
                break

            # save model, training info
            if self.enable_save and self.current_episode % self.save_freq == 0:
                str_name_save = self.save_dir + "/model_" + str(self.current_episode) + ".pth"
                torch.save(self.agent.model.state_dict(), str_name_save)
                average_length = self.summary.get_average('length')
                print(f"epsilon : {self.epsilon}, average_length : {average_length}")
                if average_length > self.max_average_length:
                    self.max_average_length = average_length
                    str_name_best = self.save_dir + "/best.pth"
                    print('best model saved - average_length: {}'.format(average_length))
                    torch.save(self.agent.model.state_dict(), str_name_best)

                self.summary.write(self.current_episode, self.epsilon)
                self.summary.clear()

            # update pbar
            pbar.update(1)

            # preview
            # if self.enable_render and self.current_episode % self.render_freq == 0:
            #     self.preview(self.render_fps)


    def preview(self, render_fps, disable_exploration=False, save_dir=None):
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        current_state = self.env.reset()

        self.env.render(fps=render_fps)
        if save_dir is not None:
            self.env.save_image(save_path=save_dir+'/0.png')

        done = False
        steps = 0
        while not done and steps < self.max_steps:
            if disable_exploration or random.random() > self.epsilon:
                action = np.argmax(self.agent.get_q_values(np.array([current_state])))
            else:
                action = np.random.randint(NUM_ACTIONS)

            next_state, reward, done = self.env.step(action)
            current_state = next_state
            steps += 1

            self.env.render(fps=render_fps)
            if save_dir is not None:
                self.env.save_image(save_path=save_dir+'/{}.png'.format(steps))

        return self.env.get_length()

    def load(self, model_filepath):
        model_filepath = self.save_dir + "/" + model_filepath + ".pth"
        self.agent.load(model_filepath)
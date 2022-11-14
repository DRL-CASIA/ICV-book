import os
import sys
import argparse
import numpy as np
import gym
import highway_env
# from random import sample
# from CarlaLCEnv import CarlaEnv, PlayGame

import matplotlib.pyplot as plt
import copy
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tensorboardX import SummaryWriter
import pickle
from PIL import Image
# from models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# hyper-parameters
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.90
EPSILON = 0.95
EPSILON_DECAY = 0.99995
MEMORY_CAPACITY = 20000
Q_NETWORK_ITERATION = 50


class ReplayBuffer(object):
    def __init__(self, max_size=MEMORY_CAPACITY):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        s, a, r, s_, d = [], [], [], [], []
        trans = transforms.ToTensor()

        for i in ind:
            state, action, reward, next_state, done = self.storage[i]
            s.append(trans(state).numpy())
            a.append(np.array(action, copy=False))
            r.append(np.array(reward, copy=False))
            s_.append(trans(next_state).numpy())
            d.append(np.array(done, copy=False))

        return np.array(s), np.array(s_), np.array(a), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class convlayer(nn.Module):
    def __init__(self):
        super(convlayer, self).__init__()
        # self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
            # nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU())
            # nn.MaxPool2d(kernel_size=2, stride=2))
        # self.drop_out = nn.Dropout()
        
    def forward(self, x):
        # print("input size: ", x.size())
        x = self.conv1(x)
        # print("conv1 size: ", x.size())
        x = self.conv2(x)
        # print("conv2 size: ", x.size())
        x = self.conv3(x)
        # print("conv3 size: ", x.size())
        # x = x.view(-1, 1152)
        # x = self.drop_out(x)
        return x


class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels, sub_sample=False, bn_layer=False):
        super(NonLocalBlock2D, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels and in_channels // 4

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU()
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=2))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # att = torch.mean(f_div_C, dim=1)
        # att = att.view(batch_size, 1, *x.size()[2:])
        # # att = F.interpolate(att, scale_factor=21.4, mode='bilinear')
        # directory = './weights_map_input_with_attention/'
        # obs_image = Image.open(directory + '/images/ob_5_0_39.png')
        # trans = transforms.ToTensor()
        # state = trans(obs_image).to('cuda')  # 
        # # att = att + state
        # att = att.squeeze(0).squeeze(0)
        # att = att.detach().cpu().numpy()
        # plt.imshow(att)
        # plt.show()
        # print('f_div_C size:', f_div_C.shape, ', g_x size:', g_x.shape, ', att size:', att.shape)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        # attention_feature = F.interpolate(y, scale_factor=21.4, mode='bilinear')
        # # print(attention_feature.shape)
        # attention_feature = torch.mean(attention_feature, dim=1)
        # # print(attention_feature.shape)
        # # print(state.shape)
        # # attention_feature = attention_feature + state
        # attention_feature = attention_feature.detach().squeeze(0).cpu().numpy()
        # plt.imshow(attention_feature)
        # plt.show()
        # # print("z size: ", z.size())
        return z


class net(nn.Module):
    def __init__(self, num_actions, dueling=False):
        super(net, self).__init__()
        # if use the dueling network
        self.dueling = dueling
        # define the network
        self.cnn_layer = convlayer()
        self.self_attention = NonLocalBlock2D(32, 32)
        # if not use dueling
        if not self.dueling:
            self.fc1 = nn.Linear(1152, 256)
            self.action_value = nn.Linear(256, num_actions)
        else:
            # the layer for dueling network architecture
            self.action_fc = nn.Linear(1152, 256)
            self.state_value_fc = nn.Linear(1152, 256)
            self.action_value = nn.Linear(256, num_actions)
            self.state_value = nn.Linear(256, 1)

    def forward(self, inputs):
        # print(inputs.size())
        x = self.cnn_layer(inputs)
        x = self.self_attention(x)
        x = x.view(-1, 1152)
        if not self.dueling:
            x = F.relu(self.fc1(x))
            action_value_out = self.action_value(x)
        else:
            # get the action value
            action_fc = F.relu(self.action_fc(x))
            action_value = self.action_value(action_fc)
            # get the state value
            state_value_fc = F.relu(self.state_value_fc(x))
            state_value = self.state_value(state_value_fc)
            # action value mean
            action_value_mean = torch.mean(action_value, dim=1, keepdim=True)
            action_value_center = action_value - action_value_mean
            # Q = V + A
            action_value_out = state_value + action_value_center
        return action_value_out


class DDQN(object):
    def __init__(self, num_actions, dueling=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_actions = num_actions
        self.epsilon = EPSILON
        self.eval_net = net(num_actions, dueling).to(self.device)
        self.target_net = net(num_actions, dueling).to(self.device)

        self.learn_step_counter = 0
        # self.memory_counter = 0
        # self.memory_counter1 = 0
        self.memory = ReplayBuffer()
        # self.memory1 = ReplayBuffer()
        self.trans = transforms.ToTensor()

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.writer = None

    def choose_action(self, state):
        # state = np.array(state)
        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state = self.trans(state).unsqueeze(0).to(self.device)
        if np.random.rand() >= self.epsilon:  # greedy policy
            action_value = self.eval_net(state).cpu()
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]
        else:  # random policy
            action = np.random.randint(0, self.num_actions)
        if self.epsilon > 0.1:
            self.epsilon *= EPSILON_DECAY
        return action

    def store_transition(self, state, action, reward, next_state, done):
        # if action == 1:
        #     self.memory1.push((state, action, reward, next_state, done))
        # else:
        #     self.memory.push((state, action, reward, next_state, done))
        self.memory.push((state, action, reward, next_state, done))

    def batch_sample(self):
        s0, s_0, a0, r0, d0 = self.memory.sample(BATCH_SIZE)
        # s1, s_1, a1, r1, d1 = self.memory1.sample(int(BATCH_SIZE / 3))
        # s = np.vstack((s0, s1))
        # a = np.vstack((a0, a1))
        # s_ = np.vstack((s_0, s_1))
        # r = np.vstack((r0, r1))
        # d = np.vstack((d0, d1))
        # return s, a, r, s_, d
        return s0, a0, r0, s_0, d0

    def learn(self):
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        s, a, r, s_, d = self.batch_sample()
        state_batch = torch.FloatTensor(s).to(self.device)
        action_batch = torch.LongTensor(a.astype(int)).view(-1, 1).to(self.device)
        next_state_batch = torch.FloatTensor(s_).to(self.device)
        reward_batch = torch.FloatTensor(r).to(self.device)
        terminal_batch = torch.FloatTensor(d).to(self.device)

        q_eval = self.eval_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            eval_next_act_batch = self.eval_net(next_state_batch).max(1)[1][:, None]
            target_next_val_batch = self.target_net(next_state_batch).gather(1, eval_next_act_batch)
        q_target = tuple(reward if terminal else reward + GAMMA * target_val for reward, terminal, target_val in
                         zip(reward_batch, terminal_batch, target_next_val_batch))
        q_target = torch.cat(q_target).view(-1, 1)

        loss = self.loss_func(q_eval, q_target)
        self.writer.add_scalar('Loss', loss, global_step=self.learn_step_counter)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def inference(self, x):
        x = self.eval_net(x)
        return x.max(1)[1]

    def save(self, directory, i):
        torch.save(self.eval_net.state_dict(), directory + 'dqn{}.pth'.format(i))
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self, directory, i):
        self.eval_net.load_state_dict(torch.load(directory + 'dqn{}.pth'.format(i)))
        print("====================================")
        print("Model has been loaded...")
        print("====================================")


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])  # mode = 'train' or 'test'
    parser.add_argument('--type', type=str, default='DDQN', help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")
    parser.add_argument('--with_PER', dest='with_per', action='store_true',
                        help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    #
    parser.add_argument('--nb_episodes', type=int, default=5000, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=1,
                        help="Number of consecutive frames (action repeat)")
    parser.add_argument('--training_interval', type=int, default=30, help="Network training frequency")
    parser.add_argument('--n_threads', type=int, default=8, help="Number of threads (A3C)")
    #
    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true',
                        help="Compute Average reward per episode (slower)")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4', help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    #
    parser.set_defaults(render=False)
    return parser.parse_args(args)


def train(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    env_config = {
                "id": "highway-v0",
                "import_module": "highway_env",
                "lanes_count": 3,
                "vehicles_count": 50,
                "duration": 50,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "observation": {
                    "type": "GrayscaleObservation",
                    "weights": [0.2989, 0.5870, 0.1140],  #weights for RGB conversion,
                    "stack_size": 1,
                    "observation_shape": (150, 600)
                },
                "screen_width": 600,  # [px]
                "screen_height": 150,  # [px]
                # "destination": "o1"
            }
    env = gym.make("highway-v0")
    env.unwrapped.configure(env_config)

    action_dim = 5
    # state_dim = 15*7
    # Pick algorithm to train
    dqn = DDQN(action_dim, dueling=True)
    directory = './weights_map_input_with_attention/'

    dqn.load(directory, 2000)
    with open(directory+'memory.pkl', 'rb') as replay_buffer:
        dqn.memory.storage = pickle.load(replay_buffer)
    dqn.epsilon = 0.10
    dqn.learn_step_counter = 58021

    dqn.writer = SummaryWriter(directory)
    episodes = 4001
    print("Collecting Experience....")
    reward_list = []
    count_image = 0
    # plt.ion()

    for i in range(2001, episodes):
        state = env.reset()
        obs = np.squeeze(state)
        img = Image.fromarray(obs.astype('uint8')).convert('L')
        state = img.resize((128, 128), Image.ANTIALIAS)
        state = state.rotate(90)
        # for _ in range(2):
        #     state, _, done = env.step(0)
        #     # plt.imshow(state)
        #     # plt.pause(0.01)
        ep_reward = 0
        for t in count():
            state.save(directory+'images/ob_{}_{}_{}.png'.format(i, t, count_image))
            count_image += 1
            action = dqn.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            obs = np.squeeze(next_state)
            img = Image.fromarray(obs.astype('uint8')).convert('L')
            next_state = img.resize((128, 128), Image.ANTIALIAS)
            next_state = next_state.rotate(90)
            # print(next_state)
            # plt.show(next_state)
            # plt.pause(0.01)

            dqn.store_transition(state, action, reward, next_state, np.float(done))
            ep_reward += reward

            memory_counter = len(dqn.memory.storage)
            # memory_counter1 = len(dqn.memory1.storage)
            # if memory_counter1 > (BATCH_SIZE/3) and memory_counter > (2*BATCH_SIZE/3):
            if memory_counter > BATCH_SIZE:
                dqn.learn()
            if done:  #  or t > 300
                dqn.writer.add_scalar('ep_r', ep_reward, global_step=i)
                print('-------------------------------------')
                print("episode: {}, the episode reward is {}".format(i, round(ep_reward, 3)))
                print("current epsilon is: {}".format(dqn.epsilon))
                print("learn step counter: {}".format(dqn.learn_step_counter))
                # print(memory_counter1, memory_counter)
                print("episode steps: {}, memory_counter: {}".format(t, memory_counter))
                break
            state = next_state
        r = copy.copy(reward)
        reward_list.append(r)
        if i % 10 == 0:
            dqn.save(directory, i)
            with open(directory+'memory.pkl', 'wb') as replay_buffer:
                pickle.dump(dqn.memory.storage, replay_buffer)
            # with open(directory+'memory.pkl', 'rb') as replay_buffer:
            #     dqn.memory.storage = pickle.load(replay_buffer)


def test(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    env_config = {
                "id": "highway-v0",
                "import_module": "highway_env",
                "lanes_count": 3,
                "vehicles_count": 50,
                "duration": 50,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "observation": {
                    "type": "GrayscaleObservation",
                    "weights": [0.2989, 0.5870, 0.1140],  #weights for RGB conversion,
                    "stack_size": 1,
                    "observation_shape": (150, 600)
                },
                "screen_width": 600,  # [px]
                "screen_height": 150,  # [px]
                # "destination": "o1"
            }
    env = gym.make("highway-v0")
    env.unwrapped.configure(env_config)

    action_dim = 5
    # state_dim = 15*7
    # Pick algorithm to train
    dqn = DDQN(action_dim, dueling=True)
    directory = './weights_map_input_with_attention/'
    dqn.epsilon = 0
    dqn.load(directory, 4000)
    for _ in range(10):
        state = env.reset()
        obs = np.squeeze(state)
        img = Image.fromarray(obs.astype('uint8')).convert('L')
        state = img.resize((128, 128), Image.ANTIALIAS)
        state = state.rotate(90)
        ep_reward = 0
        lane_change = 0
        for t in count():
            action = dqn.choose_action(state)
            if action in [0, 2]:
                lane_change += 1
            next_state, reward, done, _ = env.step(action)
            obs = np.squeeze(next_state)
            img = Image.fromarray(obs.astype('uint8')).convert('L')
            next_state = img.resize((128, 128), Image.ANTIALIAS)
            next_state = next_state.rotate(90)
            ep_reward += reward
            if done:
                print("step: {}, ep_reward: {}".format(t, ep_reward))
                with open(directory+'result4000.txt', 'a') as result:
                    result.write("step: {}, ep_reward: {}, lane change: {}".format(t, ep_reward, lane_change))
                    result.write('\n')
                break
            state = next_state
            env.render()


def test_attention():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self_attention_net = net(5, dueling=True).to(device)
    directory = './weights_map_input_with_attention/'
    i = 2000
    self_attention_net.load_state_dict(torch.load(directory + 'dqn{}.pth'.format(i)))
    self_attention_net.eval()
    obs_image = Image.open(directory + '/images/ob_5_0_39.png')
    trans = transforms.ToTensor()
    state = trans(obs_image).unsqueeze(0).to(device)
    cnn_feature = self_attention_net.cnn_layer(state)
    attention_feature = self_attention_net.self_attention(cnn_feature)
    cnn_feature = F.interpolate(cnn_feature, scale_factor=21.4, mode='bilinear')
    cnn_feature = cnn_feature + state
    # print(state.shape, cnn_feature.shape)
    cnn_feature = torch.mean(cnn_feature, dim=1).detach().squeeze(0).cpu().numpy()
    attention_feature = F.interpolate(attention_feature, scale_factor=21.4, mode='bilinear')
    attention_feature = attention_feature + state
    attention_feature = torch.mean(attention_feature, dim=1).detach().squeeze(0).cpu().numpy()
    # print(cnn_feature.shape)
    # cnn_img = Image.fromarray(cnn_feature.astype('uint8')).convert('L')
    # attention_img = Image.fromarray(attention_feature.astype('uint8')).convert('L')
    # cnn_img.save('cnn.png')
    # attention_img.save('attention.png')

    # fig = plt.figure()
    # # imgplot = plt.imshow(attention_feature)
    # cnn_plot = plt.subplot(1, 2, 1, xticks=[], yticks=[])
    # cnn_plot.set_title('cnn')
    # plt.imshow(cnn_feature)
    # attention_plot = plt.subplot(1, 2, 2, xticks=[], yticks=[])
    # attention_plot.set_title('attention')
    # plt.imshow(attention_feature)
    # plt.show()


if __name__ == "__main__":
    test()

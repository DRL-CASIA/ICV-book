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
from models import *

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
        s0, s1, a, r, s0_, s1_, d = [], [], [], [], [], [], []
        trans = transforms.ToTensor()

        for i in ind:
            state, action, reward, next_state, done = self.storage[i]
            s0.append(np.array(state['kinematics'], copy=False))
            s1.append(trans(state['image']).numpy())
            a.append(np.array(action, copy=False))
            r.append(np.array(reward, copy=False))
            s0_.append(np.array(next_state['kinematics'], copy=False))
            s1_.append(trans(next_state['image']).numpy())
            d.append(np.array(done, copy=False))

        return np.array(s0), np.array(s1), np.array(s0_), np.array(s1_), np.array(a), \
                np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


# class Net(nn.Module):
#     def __init__(self, num_states, num_actions, dueling=False):
#         super(Net, self).__init__()
#         self.dueling = dueling
#         self.fc1 = nn.Linear(num_states, 128)
#         self.fc1.weight.data.normal_(0, 0.1)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc2.weight.data.normal_(0, 0.1)
#         self.out = nn.Linear(64, num_actions)
#         self.out.weight.data.normal_(0, 0.1)
#         if self.dueling:
#             self.value = nn.Linear(64, 1)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         action_prob = self.out(x)
#         if self.dueling:
#             values = self.value(x)
#             qvals = values + (action_prob - action_prob.mean())
#             return qvals
#         return action_prob


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

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        # print("y size: ", y.size())
        W_y = self.W(y)
        # print("W_y size: ", W_y.size())
        z = W_y + x
        # attention_feature = F.interpolate(y, scale_factor=21.4, mode='bilinear')
        # attention_feature = torch.mean(attention_feature, dim=1).detach().squeeze(0).cpu().numpy()
        # plt.imshow(attention_feature)
        # plt.show()
        # print("z size: ", z.size())
        return z


class NonLocalBlock1D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=1, sub_sample=True, bn_layer=True):
        super(NonLocalBlock1D, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv1d
        max_pool_layer = nn.MaxPool1d(kernel_size=(2))
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

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


class Network(BaseModule, Configurable):
    def __init__(self, config, num_actions, dueling=True):
        super().__init__()
        Configurable.__init__(self, config)
        self.config = config
        if not self.config["embedding_layer"]["in"]:
            self.config["embedding_layer"]["in"] = self.config["in"]
        if not self.config["others_embedding_layer"]["in"]:
            self.config["others_embedding_layer"]["in"] = self.config["in"]
        self.config["output_layer"]["in"] = self.config["attention_layer"]["feature_size"]
        self.config["output_layer"]["out"] = self.config["out"]

        self.ego_embedding = model_factory(self.config["embedding_layer"])
        self.others_embedding = model_factory(self.config["others_embedding_layer"])
        self.self_attention_layer = None
        if self.config["self_attention_layer"]:
            self.self_attention_layer = SelfAttention(self.config["self_attention_layer"])
        self.attention_layer = EgoAttention(self.config["attention_layer"])
        # self.output_layer = model_factory(self.config["output_layer"])
        self.kine_att_embedding = nn.Linear(64, 256)

        self.cnn_layer = convlayer()
        self.self_attention = NonLocalBlock2D(32, 32)
        self.map_att_embedding = nn.Linear(1152, 256)

        self.dueling = dueling
        if not self.dueling:
            self.fc1 = nn.Linear(512, 64)
            self.action_value = nn.Linear(64, num_actions)
        else:
            # the layer for dueling network architecture
            self.action_fc = nn.Linear(512, 64)
            self.state_value_fc = nn.Linear(512, 64)
            self.action_value = nn.Linear(64, num_actions)
            self.state_value = nn.Linear(64, 1)

    @classmethod
    def default_config(cls):
        return {
            "in": None,
            "out": None,
            "presence_feature_idx": 0,
            "embedding_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
            "others_embedding_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
            "self_attention_layer": {
                "type": "SelfAttention",
                "feature_size": 128,
                "heads": 4
            },
            "attention_layer": {
                "type": "EgoAttention",
                "feature_size": 128,
                "heads": 4
            },
            "output_layer": {
                "type": "MultiLayerPerceptron",
                "layers": [128, 128, 128],
                "reshape": False
            },
        }

    def forward(self, kine, image):
        ego_embedded_att, _ = self.forward_attention(kine)
        kine_att_embedding = F.relu(self.kine_att_embedding(ego_embedded_att))

        x = self.cnn_layer(image)
        x = self.self_attention(x)
        x = x.view(-1, 1152)
        map_att_embedding = F.relu(self.map_att_embedding(x))
        x = torch.cat((kine_att_embedding, map_att_embedding), 1)

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

    def split_input(self, x, mask=None):
        # Dims: batch, entities, features
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        ego = x[:, 0:1, :]
        others = x[:, 1:, :]
        if mask is None:
            mask = x[:, :, self.config["presence_feature_idx"]:self.config["presence_feature_idx"] + 1] < 0.5
        return ego, others, mask

    def forward_attention(self, x):
        ego, others, mask = self.split_input(x)
        ego, others = self.ego_embedding(ego), self.others_embedding(others)
        if self.self_attention_layer:
            self_att, _ = self.self_attention_layer(ego, others, mask)
            ego, others, mask = self.split_input(self_att, mask=mask)
        return self.attention_layer(ego, others, mask)

    def get_attention_matrix(self, x):
        _, attention_matrix = self.forward_attention(x)
        return attention_matrix


class DDQN(object):
    def __init__(self, num_actions, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = EPSILON
        self.eval_net = Network(config, num_actions, dueling=True).to(self.device)
        self.target_net = Network(config, num_actions, dueling=True).to(self.device)

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
        kine = state['kinematics']
        image = state['image']
        kine = np.array(kine)
        kine = torch.FloatTensor(kine).to(self.device)
        image = self.trans(image).unsqueeze(0).to(self.device)
        if np.random.rand() >= self.epsilon:  # greedy policy
            action_value = self.eval_net(kine, image).cpu()
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
        s0, s1, s0_, s1_, a0, r0, d0 = self.memory.sample(BATCH_SIZE)
        # s1, s_1, a1, r1, d1 = self.memory1.sample(int(BATCH_SIZE / 3))
        # s = np.vstack((s0, s1))
        # a = np.vstack((a0, a1))
        # s_ = np.vstack((s_0, s_1))
        # r = np.vstack((r0, r1))
        # d = np.vstack((d0, d1))
        # return s, a, r, s_, d
        return s0, s1, a0, r0, s0_, s1_, d0

    def learn(self):
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        s0, s1, a, r, s0_, s1_, d = self.batch_sample()
        kine_batch = torch.FloatTensor(s0).to(self.device)
        image_batch = torch.FloatTensor(s1).to(self.device)
        action_batch = torch.LongTensor(a.astype(int)).view(-1, 1).to(self.device)
        next_kine_batch = torch.FloatTensor(s0_).to(self.device)
        next_image_batch = torch.FloatTensor(s1_).to(self.device)
        reward_batch = torch.FloatTensor(r).to(self.device)
        terminal_batch = torch.FloatTensor(d).to(self.device)

        q_eval = self.eval_net(kine_batch, image_batch).gather(1, action_batch)
        with torch.no_grad():
            eval_next_act_batch = self.eval_net(next_kine_batch, next_image_batch).max(1)[1][:, None]
            target_next_val_batch = self.target_net(next_kine_batch, next_image_batch).gather(1, eval_next_act_batch)
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

    # Check if a GPU ID was set
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # world, client = PlayGame.setup_world(host='localhost', fixed_delta_seconds=0.05, reload=True)
    # # client.set_timeout(5.0)
    # if world is None:
    #     return
    # traffic_manager = client.get_trafficmanager(8000)
    env_config = {
                "id": "highway-v0",
                "import_module": "highway_env",
                "lanes_count": 3,
                "vehicles_count": 50,
                "duration": 50,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "observation": {
                    "type": "GrayscaleAndKinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    # "features_range": {
                    #     "x": [-100, 100],
                    #     "y": [-100, 100],
                    #     "vx": [-20, 20],
                    #     "vy": [-20, 20]
                    # },
                    # "absolute": True,
                    # "order": "shuffled"
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
    # Pick algorithm to train
    model_config = {
        "type": "EgoAttentionNetwork",
        "embedding_layer": {
            "type": "MultiLayerPerceptron",
            "layers": [64, 64],
            "reshape": False,
            "in": 7
        },
        "others_embedding_layer": {
            "type": "MultiLayerPerceptron",
            "layers": [64, 64],
            "reshape": False,
            "in": 7
        },
        "self_attention_layer": None,
        "attention_layer": {
            "type": "EgoAttention",
            "feature_size": 64,
            "heads": 2
        },
        "output_layer": {
            "type": "MultiLayerPerceptron",
            "layers": [64, 64],
            "reshape": False
        },
        "out": 5
    }
    dqn = DDQN(action_dim, model_config)
    directory = './weights_with_two_inputs/'

    # dqn.load(directory, 2000)
    # with open(directory+'memory.pkl', 'rb') as replay_buffer:
    #     dqn.memory.storage = pickle.load(replay_buffer)
    # dqn.epsilon = 0.1
    # dqn.learn_step_counter = 56886

    dqn.writer = SummaryWriter(directory)
    episodes = 4001
    print("Collecting Experience....")
    reward_list = []
    # count_image = 56950
    # plt.ion()

    for i in range(episodes):
        state = env.reset()
        kine = state['kinematics']
        obs = np.squeeze(state['image'])
        img = Image.fromarray(obs.astype('uint8')).convert('L')
        obs = img.resize((128, 128), Image.ANTIALIAS)
        obs = obs.rotate(90)
        state = {'kinematics': kine, 'image': obs}
        ep_reward = 0
        for t in count():
            state['image'].save(directory+'images/ob_{}_{}_{}.png'.format(i, t, count_image))
            count_image += 1
            action = dqn.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            kine = next_state['kinematics']
            obs = np.squeeze(next_state['image'])
            img = Image.fromarray(obs.astype('uint8')).convert('L')
            obs = img.resize((128, 128), Image.ANTIALIAS)
            obs = obs.rotate(90)
            next_state = {'kinematics': kine, 'image': obs}
            env.render()
            # next_state = next_state.flatten()
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
        # if i % 50 == 0:
        #     world, client = PlayGame.setup_world(host='localhost', fixed_delta_seconds=0.05, reload=True)
        #     traffic_manager = client.get_trafficmanager(8000)
        #     env = CarlaEnv(world, traffic_manager)
        # ax.set_xlim(0, 300)
        # # ax.cla()
        # ax.plot(reward_list, 'g-', label='total_loss')


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
                    "type": "GrayscaleAndKinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    # "features_range": {
                    #     "x": [-100, 100],
                    #     "y": [-100, 100],
                    #     "vx": [-20, 20],
                    #     "vy": [-20, 20]
                    # },
                    # "absolute": True,
                    # "order": "shuffled"
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
    state_dim = 15*7
    # Pick algorithm to train
    model_config = {
        "type": "EgoAttentionNetwork",
        "embedding_layer": {
            "type": "MultiLayerPerceptron",
            "layers": [64, 64],
            "reshape": False,
            "in": 7
        },
        "others_embedding_layer": {
            "type": "MultiLayerPerceptron",
            "layers": [64, 64],
            "reshape": False,
            "in": 7
        },
        "self_attention_layer": None,
        "attention_layer": {
            "type": "EgoAttention",
            "feature_size": 64,
            "heads": 2
        },
        "output_layer": {
            "type": "MultiLayerPerceptron",
            "layers": [64, 64],
            "reshape": False
        },
        "out": 5
    }
    dqn = DDQN(action_dim, model_config)
    directory = './weights_with_two_inputs/'
    dqn.epsilon = 0
    dqn.load(directory, 4000)
    for _ in range(100):
        state = env.reset()
        kine = state['kinematics']
        obs = np.squeeze(state['image'])
        img = Image.fromarray(obs.astype('uint8')).convert('L')
        obs = img.resize((128, 128), Image.ANTIALIAS)
        obs = obs.rotate(90)
        state = {'kinematics': kine, 'image': obs}
        ep_reward = 0
        algo_lane_change = 0
        env_lane_change = 0
        for t in count():
            action = dqn.choose_action(state)
            if action in [0, 2]:
                algo_lane_change += 1
            next_state, reward, done, _ = env.step(action)
            kine = next_state['kinematics']
            obs = np.squeeze(next_state['image'])
            img = Image.fromarray(obs.astype('uint8')).convert('L')
            obs = img.resize((128, 128), Image.ANTIALIAS)
            obs = obs.rotate(90)
            next_state = {'kinematics': kine, 'image': obs}
            if abs(state['kinematics'][0][2] - next_state['kinematics'][0][2]) > 0.2:
                env_lane_change += 1
            ep_reward += reward
            if done:
                print("step: {}, ep_reward: {}".format(t, ep_reward))
                with open(directory+'lane_change.txt', 'a') as result:
                    result.write("step: {}, ep_reward: {}, ".format(t, ep_reward))
                    result.write("algo lane change: {}, env lane change: {}".format(algo_lane_change, env_lane_change))
                    result.write('\n')
                break
            state = next_state
            env.render()


def test_env(args=None):

    env_config = {
                "id": "highway-v0",
                "import_module": "highway_env",
                "lanes_count": 3,
                "vehicles_count": 50,
                "duration": 50,
                "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
                "observation": {
                    "type": "GrayscaleAndKinematics",
                    "vehicles_count": 15,
                    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                    # "features_range": {
                    #     "x": [-100, 100],
                    #     "y": [-100, 100],
                    #     "vx": [-20, 20],
                    #     "vy": [-20, 20]
                    # },
                    # "absolute": True,
                    # "order": "shuffled"
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
    env.reset()

    done = False
    while not done:
        action = np.random.randint(5)
        obs, reward, done, _ = env.step(action)
        # print(obs['kinematics'].shape)
        img = np.squeeze(obs['image'])
        img = Image.fromarray(img.astype('uint8')).convert('L')
        plt.pause(0.01)
        plt.imshow(img)
        env.render()


if __name__ == "__main__":
    train()

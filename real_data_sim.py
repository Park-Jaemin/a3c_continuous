from __future__ import division
from gym.spaces import Box
import numpy as np
from scipy.stats import norm
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init_mlp
from scipy import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SmartmeterEnv():
    @staticmethod
    def H(p):
        p_support = p[p.nonzero()]
        return sum(p_support * np.log2(np.reciprocal(p_support)))

    def __init__(self, x_dim=50, s_dim=50, y_dim=50, horizon=96):
        self.x_dim = x_dim
        self.s_dim = s_dim
        self.y_dim = y_dim
        self.horizon = horizon
        self.state = np.zeros(x_dim * s_dim + 1)
        self.s = 0

        # set px_x
        std = 1
        self.px_x = np.zeros((self.x_dim, self.x_dim))
        # px_x follows gaussian distribution
        for i in range(self.x_dim):
            # i means cur state
            for j in range(self.x_dim):
                # j means next state
                self.px_x[j, i] = norm.pdf(j, i, std)
            self.px_x[:, i] = self.px_x[:, i] / sum(self.px_x[:, i])

        # set pz2_yz
        px2s2_yxs = np.zeros((self.x_dim, self.s_dim, self.y_dim, self.x_dim, self.s_dim))
        for s2, y, x1, s1 in product(range(self.s_dim), range(self.y_dim), range(self.x_dim), range(self.s_dim)):
            if s2 == s1 - x1 + y:
                px2s2_yxs[:, s2, y, x1, s1] = self.px_x[:, x1]
        self.pz2_yz = px2s2_yxs.reshape(self.x_dim * self.y_dim, self.y_dim, self.x_dim * self.s_dim)

    @property
    def observation_space(self):
        """
        state[0:-1]: p(x,s)
        state[-1]: time index
        """
        observation_space = self.x_dim * self.s_dim + 1
        return Box(low=0, high=self.horizon, shape=(observation_space,), dtype=float)

    @property
    def action_space(self):
        """
        action[0,-1]: p(y|x,s)
        p(Y=y_dim | x,s) is determined by p(Y!=y_dim | x,s)
        """
        return Box(low=0, high=1, shape=((self.y_dim - 1) * self.x_dim * self.s_dim,), dtype=float)

    @property
    def reward_range(self):
        return Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    @property
    def metadata(self):
        return {'render.modes': ['human']}

    def reset(self):
        """
        state[0:-1]: p(x,s)
        state[-1]: time index

        """
        self.state = np.zeros(self.observation_space.shape[0])
        self.state[0:-1] = 1 / (self.state.shape[0] - 1)
        self.state[-1] = 1  # time step starts with 1
        return self.state

    def _valid_action(self, action):
        # change action to satisfy s+ = s - x + y
        action[action < 0] = 0
        action_mat = np.ones((self.y_dim, self.x_dim, self.s_dim))
        action_mat[0:-1, :, :] = action.reshape((self.y_dim - 1, self.x_dim, self.s_dim))
        for x, s in product(range(self.x_dim), range(self.s_dim)):
            action_mat[:, x, s] = action_mat[:, x, s] / np.sum(action_mat[:, x, s])
        for x, s, y in product(range(self.x_dim), range(self.s_dim), range(self.y_dim)):
            if s - x + y < 0:
                action_mat[x - s, x, s] += action_mat[y, x, s]
                action_mat[y, x, s] = 0
            elif s - x + y > self.s_dim - 1:
                action_mat[self.s_dim - 1 - s + x, x, s] += action_mat[y, x, s]
                action_mat[y, x, s] = 0
        return action_mat.reshape(self.y_dim * self.x_dim * self.s_dim)

    def step(self, x, action: np.ndarray):
        if np.isnan(action).any():
            print("found nan in action!")
        action = self._valid_action(action)
        pxs = self.state[0:-1]
        py_z = action.reshape((self.y_dim, self.x_dim * self.s_dim))
        pyz = py_z * pxs
        py = np.sum(pyz, axis=1)

        # get y realization
        action = np.array(action.reshape(self.y_dim, self.x_dim, self.s_dim))
        y = np.random.choice(self.y_dim, p=action[:, x, self.s])
        self.s = self.s - x + y

        # get cost
        mutual_info = self.H(py) + self.H(pxs) - self.H(pyz)
        time_cost = 61.6
        if (33 <= self.state[-1] <= 88):
            if (37 <= self.state[-1] <= 44 or 49 <= self.state[-1] <= 64):
                time_cost = 114.8
            else:
                time_cost = 84.1
        time_cost = time_cost * y
        # get next state
        pz2yz = self.pz2_yz * pyz
        pz2y = np.sum(pz2yz, axis=2)
        pz2_y = pz2y / py
        next_state = pz2_y[:, y]

        self.state[-1] = self.state[-1] + 1
        next_observation = np.zeros(self.observation_space.shape[0])
        next_observation[0:-1] = next_state
        next_observation[-1] = self.state[-1]

        reward = -mutual_info - time_cost / 10
        done = True if self.state[-1] > self.horizon else False
        meta_data = {'y': y, 'mutual_info': mutual_info, 'time_cost': time_cost}
        return next_observation, reward, done, meta_data


class A3C_MLP(torch.nn.Module):
    def __init__(self, num_inputs, action_space, n_frames):
        super(A3C_MLP, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 256)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(256, 256)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(256, 128)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.fc4 = nn.Linear(128, 128)
        self.lrelu4 = nn.LeakyReLU(0.1)

        self.m1 = n_frames * 128
        self.lstm = nn.LSTMCell(self.m1, 128)
        num_outputs = action_space.shape[0]
        self.critic_linear = nn.Linear(128, 1)
        self.actor_linear = nn.Linear(128, num_outputs)
        self.actor_linear2 = nn.Linear(128, num_outputs)

        self.apply(weights_init_mlp)
        lrelu = nn.init.calculate_gain('leaky_relu')
        self.fc1.weight.data.mul_(lrelu)
        self.fc2.weight.data.mul_(lrelu)
        self.fc3.weight.data.mul_(lrelu)
        self.fc4.weight.data.mul_(lrelu)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.train()

    def forward(self, inputs):
        x = inputs

        x = self.lrelu1(self.fc1(x))
        x = self.lrelu2(self.fc2(x))
        x = self.lrelu3(self.fc3(x))
        x = self.lrelu4(self.fc4(x))

        return F.softsign(self.actor_linear(x))


if __name__ == "__main__":
    mat_file = io.loadmat('lg data.mat')
    data = mat_file['data']
    data = np.round(data / data.max() * 40) - 1
    env = SmartmeterEnv()
    model = A3C_MLP(
        env.observation_space.shape[0], env.action_space, 1
    )
    model.load_state_dict(torch.load('trained_models/smartmeter-v0.dat'))
    observation = env.reset()
    y_arr = np.zeros(data.size)
    total_reward = 0
    total_mutual_info = 0
    total_time_cost = 0

    for i in range(data.size):
        d = data[i]
        observation = torch.from_numpy(observation).float()
        action = model.forward(observation)
        action = action.detach().numpy()
        next_observation, reward, done, meta_data = env.step(int(d[0]), action)
        total_reward = total_reward + reward
        observation = next_observation
        y = meta_data['y']
        mutual_info = meta_data['mutual_info']
        total_mutual_info = total_mutual_info + mutual_info
        time_cost = meta_data['time_cost']
        total_time_cost = total_time_cost + time_cost
        print('y: ', y)
        y_arr[i] = y

    fig = plt.figure()
    plt.plot(y_arr)
    fig.savefig('result_fig/y.png')

    fig2 = plt.figure()
    plt.plot(data)
    fig2.savefig('result_fig/x.png')

    print('mutual_info: ', total_mutual_info)
    print('time_cost: ', total_time_cost)
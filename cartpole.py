import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

epsilon = 0.4
GAMMA = 0.9
BATCH_SIZE = 128
episode_num = 10000
capacity = 1000
TARGET_UPDATE = 50
env = gym.make('FrozenLake-v0')
action_num = env.action_space.n
state_num = env.observation_space.n


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def one_hot_vec(dimension, index):
    v = torch.zeros(dimension)
    v[index] = 1
    return v


class DQN(nn.Module):

    def __init__(self, state_num, action_num):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(state_num, 8)
        self.linear2 = nn.Linear(8, action_num)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

def QLearning():
    qtable = np.random.rand(state_num, action_num)
    def SelectAction(state):
        p = random.random()
        if p > epsilon:
            return np.argmax(qtable[state])
        else:
            return env.action_space.sample()

    def Evaluate(r):
        succ_num = 0
        for i in range(1000):
            state = env.reset()
            done = False
            while not done:
                if r:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(qtable[state])
                next_state, reward, done, _ = env.step(action)
                state = next_state
            if state == 15:
                succ_num += 1
        print('success rate:', 100.0 * succ_num / 1000)
    for e in range(episode_num):
        if e % 1000 == 0:
            print(e)
        state = env.reset()
        done = False
        while not done:
            action = SelectAction(state)
            next_state, reward, done, _ = env.step(action)
            
            if done:
                qtable[state,action] += 0.1 * (reward - qtable[state,action])
            else:
                qtable[state,action] += 0.1 * (reward + GAMMA * np.max(qtable[next_state]) - qtable[state, action])
                
            state = next_state
    print(qtable)
    Evaluate(True)
            

def DeepQLearning():

    capacity = 1000

    model = DQN(state_num, action_num)
    ref = DQN(state_num, action_num)
    ref.load_state_dict(model.state_dict())
    ref.eval()

    #optimizer = optim.RMSprop(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    def SelectAction(state):
        global epsilon
        p = random.random()
        if p < epsilon:
            #            return one_hot_vec(env.action_space.sample())
            return env.action_space.sample()
        else:
            #            return one_hot_vec(model(state).max(1)[1].item())
            state_vec = one_hot_vec(state_num, state)
            return model(state_vec).argmax().item()

    memory = ReplayMemory(capacity)
    loss = nn.MSELoss()
    def Optimize():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        #print('batch.state', batch.state)
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action, dtype=torch.int64)
        reward_batch = torch.stack(batch.reward)
        mask = []
        non_final_next_state = []
        for s in batch.next_state:
            if s is not None:
                mask.append(1)
                non_final_next_state.append(s)
            else:
                mask.append(0)
        next_state_mask = torch.tensor(mask, dtype=torch.int64)
        next_state_batch = torch.stack(non_final_next_state)

        #print('state_batch', state_batch.size())
        y = model(state_batch)
        #print('y', y.size())
        #print('action_batch', action_batch.size())
        state_action_values = torch.gather(y, 1, action_batch)
        #state_action_values = model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, 1)
        y = ref(next_state_batch)
        #print('y', y.size())
        #print('next_state_values', next_state_values.size())
        next_state_values[next_state_mask] = ref(next_state_batch).max(1, keepdim=True)[0].detach()
        #next_state_value = ref(state_batch).max(1)[0].detach()
        #print('next_state_value', next_state_value.size())
        expected_state_action_values = reward_batch + (next_state_values * GAMMA)

        #loss = F.MSELoss(state_action_values, expected_state_action_values)
        #print(next_state_values.size(), reward_batch.size())
        #print(state_action_values.size(), expected_state_action_values.size())
        output = loss(state_action_values, expected_state_action_values)
        optimizer.zero_grad()
        output.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def Evaluate():
        success_num = 0
        for i in range(50):
            state = env.reset()
            done = False
            while not done:
                state_vec = one_hot_vec(state_num, state)
                action = model(state_vec).argmax().item()
                state, _, done, _ = env.step(action)
            if state == 15:
                success_num += 1
        print("Success ratio: ", 1.0 * success_num / 50 * 100, '%')

    for e in range(episode_num):
        state = env.reset()
        done = False
        #while not done:
        global epsilon
        for step in count():
            if step > 0 and step % 1000 == 0:
                epsilon -= 0.099
            action = SelectAction(state)
            next_state, reward, done, _ = env.step(action)

            state_vec = one_hot_vec(state_num, state)
            next_state_vec = one_hot_vec(state_num, next_state)
            reward_vec = torch.Tensor([reward])

            if done:
                next_state = None
            memory.push(state_vec, [action], next_state_vec, reward_vec)

            state = next_state

            Optimize()

            if done:
                break
        if e % TARGET_UPDATE == 0:
            ref.load_state_dict(model.state_dict())
            Evaluate()



if __name__ == '__main__':
    #DeepQLearning()
    QLearning()

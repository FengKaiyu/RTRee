import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as f
import torch.multiprocessing as mp
import torch.optim as optim

import random
import math
import argparse
import pickle
from collections import namedtuple
from itertools import count
import os
import time
from datetime import datetime

from RTree import RTree

parser = argparse.ArgumentParser()

parser.add_argument('-action', choices=['train', 'test', 'baseline'])
parser.add_argument('-objfile', help='data file')
parser.add_argument('-queryfile', help='query file')
parser.add_argument('-epoch', type=int, help='number of epoches')
parser.add_argument('-reward_query_width', type=float, default=0.01)
parser.add_argument('-reward_query_height', type=float, default=0.01)
parser.add_argument('-default_ins_strategy', help='default insert strategy', default="INS_AREA")
parser.add_argument('-default_spl_strategy', help='default split strategy', default='SPL_MIN_AREA')
parser.add_argument('-reference_tree_ins_strategy', help='default insert strategy for reference tree', default="INS_AREA")
parser.add_argument('-reference_tree_spl_strategy', help='default split strategy for reference tree', default='SPL_MIN_AREA')
parser.add_argument('-action_space', type=int, help='number of possible actions', default=5)
parser.add_argument('-batch_size', type=int, help='batch_size', default=64)
parser.add_argument('-state_dim', type=int, help='input dimension', default=25)
parser.add_argument('-inter_dim', type=int, help='internal dimension', default=32)
parser.add_argument('-memory_cap', type=int, help='memory capacity', default=5000)
parser.add_argument('-lr', type=float, help='learning rate', default=0.01)
parser.add_argument('-gamma', type=float, help='reward discount factor', default=0.8)
parser.add_argument('-model_name', help='name of the model')
parser.add_argument('-target_update', type=int, help='update the parameters for target network every ? steps', default=30)
parser.add_argument('-epsilon', type=float, help='epsilon greedy', default=0.9)
parser.add_argument('-epsilon_decay', type=float, help='how fast to decrease epsilon', default=0.99)
parser.add_argument('-min_epsilon', type=float, help='minimum epsilon', default=0.1)
parser.add_argument('-max_entry', type=int, help='maximum entry a node can hold', default=100)
parser.add_argument('-query_for_reward', type=int, help='number of query used for reward', default=5)

class DQN(nn.Module):
	def __init__(self, input_dimension, inter_dimension, output_dimension):
		super(DQN, self).__init__()
		self.linear1 = nn.Linear(input_dimension, inter_dimension)
		self.linear2 = nn.Linear(inter_dimension, output_dimension)

	def forward(self, x):
		m = nn.SELU()
		x = self.linear1(x)
		x = m(x)
		x = self.linear2(x)
		#sf = nn.Softmax(dim=0)
		return x

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

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

	def clear(self):
		self.memory.clear()
		self.position = 0

	def __len__(self):
		return len(self.memory)

class SplitLearner:
	def __init__(self):
		self.tree = None
		self.reference_tree = None
		self.network = None
		self.target_network = None
		self.memory = None
		self.obj_input = None
		self.query_input = None
		self.config = None

	def Initialize(self, config):
		self.config = config
		if config.objfile:
			self.obj_input = open(config.objfile, 'r')
		if config.queryfile:
			self.query_input = open(config.queryfile, 'r')

		self.tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
		self.tree.SetInsertStrategy(config.default_ins_strategy)
		self.tree.SetSplitStrategy(config.default_spl_strategy)
		self.reference_tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
		self.reference_tree.SetInsertStrategy(config.reference_tree_ins_strategy)
		self.reference_tree.SetSplitStrategy(config.reference_tree_spl_strategy)

		self.network = DQN(self.config.state_dim, self.config.inter_dim, self.config.action_space)
		self.target_network = DQN(self.config.state_dim, self.config.inter_dim, self.config.action_space)

		self.target_network.load_state_dict(self.network.state_dict())
		self.target_network.eval()

		self.memory = ReplayMemory(self.config.memory_cap)
		self.loss = nn.MSELoss()
		self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.config.lr)


	def NextObj(self):
		line = self.obj_input.readline()
		if not line:
			return None
		boundary = [float(e) for e in line.strip().split()]
		return boundary

	def ResetObjLoader(self):
		self.obj_input.seek(0)

	def NextQuery(self):
		line = self.query_input.readline()
		if not line:
			return None
		boundary = [float(e) for e in line.strip().split()]
		return boundary

	def EpsilonGreedy(self, q_values):
		p = random.random()
		if p < self.config.epsilon:
			return torch.tensor(random.randint(0, self.config.action_space - 1), dtype=torch.int64)
		else:
			return np.argmax(q_values)

	def Optimize(self):
		if len(self.memory) < self.config.batch_size:
			return

		transitions = self.memory.sample(self.config.batch_size)
		batch = Transition(*zip(*transitions))

		state_batch = torch.stack(batch.state)
		reward_batch = torch.stack(batch.reward)
		action_batch = torch.unsqueeze(torch.stack(batch.action), 1)
		

		real_q_values = self.network(state_batch)

		state_action_values = torch.gather(real_q_values, 1, action_batch)

		mask = []
		non_final_next_state = []
		for s in batch.next_state:
			if s is not None:
				mask.append(1)
				non_final_next_state.append(s)
			else:
				mask.append(0)
		next_state_values = torch.zeros(self.config.batch_size, 1)
		if non_final_next_state:
			next_state_mask = torch.nonzero(torch.tensor(mask, dtype=torch.int64)).squeeze(1)
			next_state_batch = torch.stack(non_final_next_state)	
			y, _ = self.target_network(next_state_batch).max(1, keepdim=True)
			next_state_values[next_state_mask] = y
		expected_state_action_values = reward_batch + (next_state_values * self.config.gamma)

		output = self.loss(state_action_values, expected_state_action_values)
		l = output.item()
		self.optimizer.zero_grad()
		output.backward()
		for param in self.network.parameters():
			param.grad.data.clamp_(-1, 1)
		self.optimizer.step()
		return l

	def ComputeReward(self):
		access_rate_avg = 0
		for i in range(self.config.query_for_reward):
			query = self.tree.UniformRandomQuery(self.config.reward_query_width, self.config.reward_query_height)
			access_rate_avg += self.tree.AccessRate(query) - self.reference_tree.AccessRate(query)
		return access_rate_avg / self.config.query_for_reward

	def ComputeDenseReward(self, object_boundary):
		access_rate_avg = 0
		for i in range(self.config.query_for_reward):
			query = self.tree.UniformDenseRandomQuery(self.config.reward_query_width, self.config.reward_query_height, object_boundary)
			access_rate_avg += self.tree.AccessRate(query) - self.reference_tree.AccessRate(query)
		return access_rate_avg / self.config.query_for_reward


	def Test(self):
		self.network.load_state_dict(torch.load("./model/"+self.config.model_name+".mdl"))
		self.network.eval()
		self.ResetObjLoader()
		self.tree.Clear()

		object_boundary = self.NextObj()
		obj_cnt = 0
		while object_boundary is not None:
			obj_cnt += 1
			if obj_cnt % 100 == 0:
				print(obj_cnt)
			self.tree.DirectInsert(object_boundary)
			#print('insert')
			states, _ = self.tree.RetrieveSplitStates()
			while states is not None:
				states = torch.tensor(states, dtype=torch.float32)
				#print('states', states)
				#input()
				q_values = self.network(states)
				#print('qvalues', qvalues)
				#input()
				action = torch.argmax(q_values).item()
				#print('action', action)
				#input()
				self.tree.SplitOneStep(action)
				states, _ = self.tree.RetrieveSplitStates()
			object_boundary = self.NextObj()

		node_access = 0
		query_num = 0
		query = self.NextQuery()
		while query is not None:
			node_access += self.tree.Query(query)
			query_num += 1
			query = self.NextQuery()
		print('average node access is ', node_access / query_num)
		return 1.0 * node_access / query_num

	def Baseline(self):
		object_boundary = self.NextObj()
		while object_boundary is not None:
			self.tree.DefaultInsert(object_boundary)
			object_boundary = self.NextObj()

		node_access = 0
		query_num = 0

		query = self.NextQuery()
		while query is not None:
			node_access += self.tree.Query(query)
			query_num += 1
			query = self.NextQuery()
		return 1.0 * node_access / query_num



	def Train(self):
		start_time = time.time()
		loss_log = open("./log/{}.loss".format(self.config.model_name), 'w')
		split_triggered = 0
		useful_split = 0
		unuseful_split = 0
		reward_is_0 = 0
		reward2_is_0 = 0
		for epoch in range(self.config.epoch):
			e = 0
			self.ResetObjLoader()
			self.tree.Clear()
			self.reference_tree.Clear()
			object_boundary = self.NextObj()
			while object_boundary is not None:

				self.reference_tree.CopyTree(self.tree.tree)
				self.reference_tree.DefaultInsert(object_boundary)
				self.tree.DirectInsert(object_boundary)
				states, is_valid = self.tree.RetrieveSplitStates()
				steps = []
				while states is not None:
					states = torch.tensor(states, dtype=torch.float32)
					with torch.no_grad():
						q_values = self.network(states)
					action = self.EpsilonGreedy(q_values)
					self.tree.SplitOneStep(action)
					steps.append((states, action, is_valid))
					states, is_valid = self.tree.RetrieveSplitStates()

				if steps:
					split_triggered += 1
					reward2 = self.ComputeReward()
					reward = self.ComputeDenseReward(object_boundary)
					if reward == 0:
						reward_is_0 += 1
					if reward2 == 0:
						reward2_is_0 += 1
					for i in range(len(steps)-1):
						if steps[i][2]:
							useful_split += 1
							self.memory.push(steps[i][0], steps[i][1], torch.tensor([reward]), steps[i+1][0])
						else:
							unuseful_split += 1
					if steps[-1][2]:
						useful_split += 1
						self.memory.push(steps[-1][0], steps[-1][1], torch.tensor([reward]), None)
					else:
						unuseful_split += 1
				l = self.Optimize()
				loss_log.write('{}\n'.format(l))
				if e % 100 == 0:
					print('{} objects added, loss is {}\n'.format(e, l))
					self.config.epsilon = max(self.config.epsilon * self.config.epsilon_decay, self.config.min_epsilon)
				e += 1

				object_boundary = self.NextObj()

				if e % self.config.target_update == 0:
					self.target_network.load_state_dict(self.network.state_dict())

			torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.epoch{}'.format(epoch)+'.mdl')

		torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.mdl')
		end_time = time.time()
		train_log = open('./log/train.log', 'a')
		train_log.write('{}:\n'.format(datetime.now()))
		train_log.write('{}\n'.format(self.config))
		train_log.write('training time: {}, {} objects triggered splitting, {} useful split, {} unuseful split\n'.format(end_time-start_time, split_triggered, useful_split, unuseful_split))
		train_log.write('zero reward: {}, zero reward2: {}\n'.format(reward_is_0, reward2_is_0))
		train_log.close()



if __name__ == '__main__':
	args = parser.parse_args()
	device = torch.device("cpu")

	if args.action == 'train':
		spl_learner = SplitLearner()
		spl_learner.Initialize(args)
		spl_learner.Train()
	if args.action == 'test':
		spl_learner = SplitLearner()
		spl_learner.Initialize(args)
		spl_learner.Test()


from tqdm import trange
import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim

import hashlib

import random
import argparse
from collections import namedtuple
import time
from datetime import datetime

from RTree import RTree

parser = argparse.ArgumentParser()

parser.add_argument('-action', choices=['train', 'test', 'test10', 'baseline'])
parser.add_argument('-objfile', help='data file')
parser.add_argument('-queryfile', help='query file')
parser.add_argument('-epoch', type=int, help='number of epoches', default=1)
parser.add_argument('-reward_query_width', type=float, default=0.01)
parser.add_argument('-reward_query_height', type=float, default=0.01)
parser.add_argument('-default_ins_strategy', help='default insert strategy', default="INS_AREA")
parser.add_argument('-default_spl_strategy', help='default split strategy', default='SPL_MIN_AREA')
parser.add_argument('-reference_tree_ins_strategy', help='default insert strategy for reference tree', default="INS_AREA")
parser.add_argument('-reference_tree_spl_strategy', help='default split strategy for reference tree', default='SPL_MIN_MARGIN')
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
parser.add_argument('-max_entry', type=int, help='maximum entry a node can hold', default=50)
parser.add_argument('-query_for_reward', type=int, help='number of query used for reward', default=5)
parser.add_argument('-splits_for_update', type=int, help='number of splits for a reward computation', default=20)
parser.add_argument('-parts', type=int, help='number of parts to train', default=5)
parser.add_argument('-network', choices=['strategy', 'spl_loc', 'spl_loc_short', 'sort_spl_loc'], help='which network is used for training', default='strategy')
parser.add_argument('-teacher_forcing', type=float, help='the percentage of splits that are with teacher forcing technique')


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
        # sf = nn.Softmax(dim=0)
        return x


class DQN2(nn.Module):
    def __init__(self, input_dimension=240, inter_dimension=300, output_dimension=48):
        super(DQN2, self).__init__()
        self.linear1 = nn.Linear(input_dimension, inter_dimension)
        self.linear2 = nn.Linear(inter_dimension, output_dimension)

    def forward(self, x):
        m = nn.SELU()
        x = self.linear1(x)
        x = m(x)
        x = self.linear2(x)
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
        md5content = "{}".format(datetime.now())
        self.id = hashlib.md5(md5content.encode()).hexdigest()

    def Initialize(self, config):
        self.config = config
        if config.objfile:
            try:
                self.obj_input = open(config.objfile, 'r')
            except:
                print('object file does not exist.')
        if config.queryfile:
            try:
                self.query_input = open(config.queryfile, 'r')
            except:
                print('query file does not exist.')

        self.tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
        self.tree.SetInsertStrategy(config.default_ins_strategy)
        self.tree.SetSplitStrategy(config.default_spl_strategy)
        self.reference_tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
        self.reference_tree.SetInsertStrategy(config.reference_tree_ins_strategy)
        self.reference_tree.SetSplitStrategy(config.reference_tree_spl_strategy)

        if self.config.network == 'strategy':
            self.network = DQN(self.config.state_dim, self.config.inter_dim, self.config.action_space)
            self.target_network = DQN(self.config.state_dim, self.config.inter_dim, self.config.action_space)
        if self.config.network == 'spl_loc':
            self.network = DQN2()
            self.target_network = DQN2()
        if self.config.network == 'spl_loc_short':
            self.network = DQN2(60, 60, 12)
            self.target_network = DQN2(60, 60, 12)
        if self.config.network == 'sort_spl_loc':
            self.network = DQN2(self.config.state_dim, self.config.inter_dim, self.config.action_space)
            self.target_network = DQN2(self.config.state_dim, self.config.inter_dim, self.config.action_space)




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

    def ResetQueryLoader(self):
        self.query_input.seek(0)

    def EpsilonGreedy(self, q_values):
        p = random.random()
        if p < self.config.epsilon:
            return torch.tensor(random.randint(0, self.config.action_space - 1), dtype=torch.int64)
        else:
            return np.argmax(q_values)

    def Optimize(self):
        if len(self.memory) < self.config.batch_size:
            return None

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
            next_state_mask = torch.nonzero(torch.tensor(mask, dtype=torch.int64), as_tuple=False).squeeze(1)
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
            access_rate_avg += self.reference_tree.AccessRate(query) - self.tree.AccessRate(query)
        return access_rate_avg / self.config.query_for_reward

    def ComputeDenseRewardForList(self, obj_list):
        access_rate_avg = 0
        for obj in obj_list:
            for i in range(5):
                query = self.tree.UniformDenseRandomQuery(self.config.reward_query_width, self.config.reward_query_height, obj)
                #print(query)
                reference_rate = self.reference_tree.AccessRate(query)
                #print(reference_rate)
                tree_rate = self.tree.AccessRate(query)
                #print(tree_rate)
                access_rate_avg += reference_rate - tree_rate
                #print(access_rate_avg)
        return access_rate_avg / len(obj_list) / 5

    def ComputeDenseReward(self, object_boundary):
        access_rate_avg = 0
        for i in range(self.config.query_for_reward):
            query = self.tree.UniformDenseRandomQuery(self.config.reward_query_width, self.config.reward_query_height, object_boundary)
            access_rate_avg += self.reference_tree.AccessRate(query) - self.tree.AccessRate(query)
        return access_rate_avg / self.config.query_for_reward


    def Test10(self):
        self.network.load_state_dict(torch.load("./model/"+self.config.model_name+".mdl"))
        self.network.eval()
        node_access = 0
        query_num = 0
        for i in range(10):
            self.tree.Clear()
            name = "./dataset/{}.test{}.txt".format(self.config.objfile, i)
            ofin = open(name, 'r')
            for line in ofin:
                object_boundary = [float(e) for e in line.strip().split()]
                self.tree.DirectInsert(object_boundary)
                states, _ = self.tree.RetrieveSplitStates()
                while states is not None:
                    states = torch.tensor(states, dtype=torch.float32)
                    q_values = self.network(states)
                    action = torch.argmax(q_values).item()
                    self.tree.SplitOneStep(action)
                    states, _ = self.tree.RetrieveSplitStates()
            ofin.close()
            query = self.NextQuery()
            while query is not None:
                node_access += self.tree.Query(query)
                query_num += 1
                query = self.NextQuery()
            self.ResetQueryLoader()
        print('average node access is ', node_access / query_num)

    def Test10_2(self):
        self.network.load_state_dict(torch.load("./model/"+self.config.model_name+".mdl"))
        self.network.eval()
        node_access = 0
        query_num = 0
        for i in range(10):
            self.tree.Clear()
            name = "./dataset/{}.test{}.txt".format(self.config.objfile, i)
            ofin = open(name, 'r')
            for line in ofin:
                object_boundary = [float(e) for e in line.strip().split()]
                self.tree.DirectInsert(object_boundary)
                states = self.tree.RetrieveSpecialSplitStates()
                while states is not None:
                    states = torch.tensor(states, dtype=torch.float32)
                    q_values = self.network(states)
                    action = torch.argmax(q_values).item()
                    self.tree.SplitWithLoc(action)
                    states = self.tree.RetrieveSpecialSplitStates()
            ofin.close()
            query = self.NextQuery()
            while query is not None:
                node_access += self.tree.Query(query)
                query_num += 1
                query = self.NextQuery()
            self.ResetQueryLoader()
        print('average node access is ', node_access / query_num)

    def Test2(self):
        self.network.load_state_dict(torch.load("./model/"+self.config.model_name+".mdl"))
        self.network.eval()
        self.ResetObjLoader()
        self.tree.Clear()
        self.reference_tree.Clear()
        object_boundary = self.NextObj()
        obj_cnt = 0
        while object_boundary is not None:
            obj_cnt += 1
            self.tree.DirectInsert(object_boundary)
            self.reference_tree.DefaultInsert(object_boundary)
            states = None
            if self.config.network == 'spl_loc':
                states = self.tree.RetrieveSpecialSplitStates()
            elif self.config.network == 'spl_loc_short':
                states = self.tree.RetrieveShortSplitStates()
            elif self.config.network == 'sort_spl_loc':
                states = self.tree.RetrieveSortedSplitStates(self.config.action_space)
            while states is not None:
                states = torch.tensor(states, dtype=torch.float32)
                q_values = self.network(states)
                action = torch.argmax(q_values).item()
                if self.config.network == 'sort_spl_loc':
                    success = self.tree.SplitWithSortedLoc(action)
                else:
                    self.tree.SplitWithLoc(action)
                if self.config.network == 'spl_loc':
                    states = self.tree.RetrieveSpecialSplitStates()
                elif self.config.network == 'spl_loc_short':
                    states = self.tree.RetrieveShortSplitStates()
                elif self.config.network == 'sort_spl_loc':
                    states = self.tree.RetrieveSortedSplitStates(self.config.action_space)
            object_boundary = self.NextObj()
        #self.tree.PrintEntryNum()
        print('average tree node area: ', self.tree.AverageNodeArea())
        print('average tree node children: ', self.tree.AverageNodeChildren())
        print('total tree nodes: ', self.tree.TotalTreeNodeNum())
        node_access = 0
        query_num = 0
        query = self.NextQuery()
        f = open('debug.result.log', 'w')
        f2 = open('reference.result.log', 'w')
        reference_node_access = 0.0
        while query is not None:
            node_access += self.tree.Query(query)
            f.write('{}\n'.format(self.tree.QueryResult()))
            reference_node_access += self.reference_tree.Query(query)
            f2.write('{}\n'.format(self.reference_tree.QueryResult()))
            query_num += 1
            query = self.NextQuery()
        print('average node access is ', node_access / query_num)
        print('reference node access is ', reference_node_access/query_num)
        f.close()
        f2.close()
        return 1.0 * node_access / query_num

    def Test(self):
        self.network.load_state_dict(torch.load("./model/"+self.config.model_name+".mdl"))
        self.network.eval()
        self.ResetObjLoader()
        self.tree.Clear()
        debug = False
        object_boundary = self.NextObj()
        obj_cnt = 0
        self.tree.debug = False
        while object_boundary is not None:
            obj_cnt += 1
            #if obj_cnt > 92300:
                #debug = True
            #if obj_cnt % 100 == 0:
                #print(obj_cnt)
            if debug:
                print('to insert', obj_cnt)
            self.tree.DirectInsert(object_boundary)
            if debug:
                print('inserted', obj_cnt)
            #print('insert')
            states, _ = self.tree.RetrieveSplitStates()
            #if debug:
            #print(states)
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
                #if(debug):
                #print("to split")
                self.tree.SplitOneStep(action)
                #if(debug):
                #    print('splitted')
                states, _ = self.tree.RetrieveSplitStates()
            object_boundary = self.NextObj()
        self.tree.PrintEntryNum()


        node_access = 0
        query_num = 0
        query = self.NextQuery()
        f = open('debug.result.log', 'w')
        while query is not None:
            node_access += self.tree.Query(query)
            f.write('{}\n'.format(self.tree.QueryResult()))
            query_num += 1
            query = self.NextQuery()
        print('average node access is ', node_access / query_num)
        f.close()
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


    def Train5(self):
        #with teacher forcing
        start_time = time.time()
        loss_log = open("./log/{}.loss".format(self.id), 'w')
        reward_log = open("./log/{}.reward".format(self.id), "w")
        steps = []
        object_num = 0
        self.ResetObjLoader()
        object_boundary = self.NextObj()
        cache_tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
        cache_tree.SetInsertStrategy(self.config.default_ins_strategy)
        cache_tree.SetSplitStrategy(self.config.default_spl_strategy)

        while object_boundary is not None:
            object_num += 1
            object_boundary = self.NextObj()
        for epoch in trange(self.config.epoch, desc='Epoch'):
            e = 0
            part_trange = trange(self.config.parts, desc='With parts')
            for part in part_trange:
                ratio_for_tree_construction = 1.0 * (part + 1) / (self.config.parts + 1)
                self.ResetObjLoader()
                self.tree.Clear()
                self.reference_tree.Clear()
                cache_tree.Clear()
                for i in range(int(object_num * ratio_for_tree_construction)):
                    object_boundary = self.NextObj()
                    self.tree.DefaultInsert(object_boundary)
#                print('{} nodes are added into the tree'.format(int(object_num * ratio_for_tree_construction)))
                part_trange.set_description('With parts: {} nodes are added into the tree'.format(int(object_num * ratio_for_tree_construction)))
                part_trange.refresh()

                fo = open('train_object.tmp', 'w')
                object_boundary = self.NextObj()
                #print('filling leaf nodes')
                objects_for_fill = 0
                objects_for_train = 0
                cnt = 0
                while object_boundary is not None:
                    cnt += 1
                    is_success = self.tree.TryInsert(object_boundary)
                    if not is_success:
                        objects_for_train += 1
                        fo.write('{:.5f} {:.5f} {:.5f} {:.5f}\n'.format(object_boundary[0], object_boundary[1], object_boundary[2], object_boundary[3]))
                    else:
                        objects_for_fill += 1
                    object_boundary = self.NextObj()
                fo.close()
                cache_tree.CopyTree(self.tree.tree)
                self.reference_tree.CopyTree(cache_tree.tree)
                self.tree.CopyTree(cache_tree.tree)

 
                fin = open('train_object.tmp', 'r')
                period = 0
                obj_list_for_reward = []
                accum_loss = 0
                accum_loss_cnt = 0
                split_trange = trange(objects_for_train, desc="Training", leave=False)
                for training_id in split_trange:
                    line = fin.readline()
                    object_boundary = [float(v) for v in line.strip().split()]
                    self.reference_tree.DefaultInsert(object_boundary)
                    self.tree.DirectInsert(object_boundary)
                    state = None
                    if self.config.network == 'spl_loc':
                        states = self.tree.RetrieveSpecialSplitStates()
                    elif self.config.network == 'spl_loc_short':
                        states = self.tree.RetrieveShortSplitStates()
                    elif self.config.network == 'sort_spl_loc':
                        states = self.tree.RetrieveSortedSplitStates(self.config.action_space)
                    trigger_split = False
                    while states is not None:
                        trigger_split = True
                        states = torch.tensor(states, dtype=torch.float32)
                        action = None
                        if self.config.teacher_forcing is not None:
                            # use teacher forcing
                            splits_with_tf = int(self.config.teacher_forcing * objects_for_train)
                            if training_id < splits_with_tf:
                                min_perimeter = 1000000000000.0
                                perimeters = np.zeros(self.config.action_space)
                                for i in range(self.config.action_space):
                                    perimeters[i] = states[i * 5 + 2] + states[i * 5 + 3]
                                perimeters = torch.tensor(perimeters, dtype=torch.float32)
                                action = np.argmin(perimeters)
                        if action == None:
                            with torch.no_grad():
                                q_values = self.network(states)
                                action = self.EpsilonGreedy(q_values)
                        if self.config.network == 'sort_spl_loc':
                            self.tree.SplitWithSortedLoc(action)
                        else:
                            self.tree.SplitWithLoc(action)
                        steps.append((states, action))
                        if self.config.network == 'spl_loc':
                            states = self.tree.RetrieveSpecialSplitStates()
                        elif self.config.network == 'spl_loc_short':
                            states = self.tree.RetrieveShortSplitStates()
                        elif self.config.network == 'sort_spl_loc':
                            states = self.tree.RetrieveSortedSplitStates(self.config.action_space)
                    
                    if trigger_split:
                        steps.append((None, None))
                    period += 1
                    obj_list_for_reward.append(object_boundary)

                    if period == self.config.splits_for_update:
                        reward = self.ComputeDenseRewardForList(obj_list_for_reward)
                        reward_log.write('{}\n'.format(reward))
                        for i in range(len(steps) - 1):
                            if steps[i][1] is None:
                                continue
                            self.memory.push(steps[i][0], steps[i][1], torch.tensor([reward]), steps[i+1][0])
                        self.reference_tree.CopyTree(cache_tree.tree)
                        self.tree.CopyTree(cache_tree.tree)
                        period = 0
                        obj_list_for_reward.clear()
                        steps.clear()
                    
                    l = self.Optimize()
                    if l is not None:
                        accum_loss += l
                        accum_loss_cnt += 1
                    loss_log.write('{}\n'.format(l))
                    if e % 500 == 0:
                        average_loss = None
                        if accum_loss > 0:
                            average_loss = 1.0 * accum_loss / accum_loss_cnt
                        split_trange.set_description('Training: average loss {}'.format(average_loss))
                        accum_loss = 0
                        accum_loss_cnt = 0
                        split_trange.refresh()
                        self.config.epsilon = max(self.config.epsilon * self.config.epsilon_decay, self.config.min_epsilon)
                    e += 1
                    if e % self.config.target_update == 0:
                        self.target_network.load_state_dict(self.network.state_dict())
                fin.close()
            torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.epoch{}'.format(epoch) + '.mdl')
        end_time = time.time()
        torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.'+self.id+'.mdl')
        reward_log.close()
        loss_log.close()
        train_log = open('./log/train.log', 'a')
        train_log.write('{}:\n'.format(datetime.now()))
        train_log.write('{}\n'.format(self.id))
        train_log.write('{}\n'.format(self.config))
        train_log.write('training time: {}\n'.format(end_time-start_time))
        #train_log.write('zero reward: {}, zero reward2: {}\n'.format(reward_is_0, reward2_is_0))
        train_log.close()
        self.tree.Clear()
        self.reference_tree.Clear()
        cache_tree.Clear()



    def Train4(self):
        # learn which location to split
        start_time = time.time()
        loss_log = open("./log/{}.loss".format(self.id), 'w')
        reward_log = open("./log/{}.reward".format(self.id), "w")
        steps = []
        object_num = 0
        self.ResetObjLoader()
        object_boundary = self.NextObj()
        cache_tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
        cache_tree.SetInsertStrategy(self.config.default_ins_strategy)
        cache_tree.SetSplitStrategy(self.config.default_spl_strategy)

        while object_boundary is not None:
            object_num += 1
            object_boundary = self.NextObj()
        objects_for_train = 0
        for epoch in range(self.config.epoch):
            e = 0
            self.ResetObjLoader()
            self.tree.Clear()
            self.reference_tree.Clear()
            print("setup initial tree")
            ratio_for_tree_construction = epoch % self.config.parts + 1
            for i in range(int(object_num * ratio_for_tree_construction / (self.config.parts + 1))):
                object_boundary = self.NextObj()
                self.tree.DefaultInsert(object_boundary)
            fo = open('train_object.tmp', 'w')
            object_boundary = self.NextObj()

            print('filling leaf nodes')
            #fill the r-tree so every leaf is full
            objects_for_fill = 0
            cnt = 0
            while object_boundary is not None:
                if cnt % 100 == 0:
                    print(cnt)
                cnt += 1
                is_success = self.tree.TryInsert(object_boundary)
                if not is_success:
                    fo.write('{:.5f} {:.5f} {:.5f} {:.5f}\n'.format(object_boundary[0], object_boundary[1], object_boundary[2], object_boundary[3]))
                else:
                    objects_for_fill += 1
                object_boundary = self.NextObj()
            fo.close()
            cache_tree.CopyTree(self.tree.tree)
            self.reference_tree.CopyTree(cache_tree.tree)
            self.tree.CopyTree(cache_tree.tree)


            fin = open('train_object.tmp', 'r')
            period = 0
            obj_list_for_reward = []
            for line in fin:
                objects_for_train += 1
                object_boundary = [float(v) for v in line.strip().split()]
                #print('object', object_boundary)
                self.reference_tree.DefaultInsert(object_boundary)
                self.tree.DirectInsert(object_boundary)
                states = self.tree.RetrieveSpecialSplitStates()
                triggered = False
                while states is not None:
                    triggered = True
                    states = torch.tensor(states, dtype=torch.float32)
                    with torch.no_grad():
                        q_values = self.network(states)
                        action = self.EpsilonGreedy(q_values)
                        self.tree.SplitWithLoc(action)
                        steps.append((states, action))
                        states = self.tree.RetrieveSpecialSplitStates()


                if triggered:
                    steps.append((None, None))
                period += 1
                obj_list_for_reward.append(object_boundary)

                if period == self.config.splits_for_update:
                    reward = self.ComputeDenseRewardForList(obj_list_for_reward)
                    reward_log.write('{}\n'.format(reward))
                    for i in range(len(steps) - 1):
                        if steps[i][1] is None:
                            continue
                        self.memory.push(steps[i][0], steps[i][1], torch.tensor([reward]), steps[i+1][0])
                    self.reference_tree.CopyTree(cache_tree.tree)
                    self.tree.CopyTree(cache_tree.tree)
                    period = 0
                    obj_list_for_reward.clear()
                    steps.clear()


                l = self.Optimize()
                loss_log.write('{}\n'.format(l))
                if e % 500 == 0:
                    print('{} objects added, loss is {}\n'.format(e, l))
                    self.config.epsilon = max(self.config.epsilon * self.config.epsilon_decay, self.config.min_epsilon)
                e += 1

                if e % self.config.target_update == 0:
                    self.target_network.load_state_dict(self.network.state_dict())
            torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.epoch{}'.format(epoch)+'.mdl')
            fin.close()
        end_time = time.time()
        torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.'+self.id+'.mdl')
        reward_log.close()
        loss_log.close()
        train_log = open('./log/train.log', 'a')
        train_log.write('{}:\n'.format(datetime.now()))
        train_log.write('{}\n'.format(self.id))
        train_log.write('{}\n'.format(self.config))
        train_log.write('training time: {}\n'.format(end_time-start_time))
        #train_log.write('zero reward: {}, zero reward2: {}\n'.format(reward_is_0, reward2_is_0))
        train_log.close()
        self.tree.Clear()
        self.reference_tree.Clear()
        cache_tree.Clear()


    def Train3(self):
        #Use the R-tree with full leaf nodes to train
        start_time = time.time()
        loss_log = open("./log/{}.loss".format(self.config.model_name), 'w')
        debug_log = open('./reward.log', 'w')
        split_triggered = 0
        useful_split = 0
        unuseful_split = 0
        reward_is_0 = 0
        reward2_is_0 = 0
        steps = []
        object_num = 0
        self.ResetObjLoader()
        object_boundary = self.NextObj()
        cache_tree = RTree(self.config.max_entry, int(0.4 * self.config.max_entry))
        cache_tree.SetInsertStrategy(self.config.default_ins_strategy)
        cache_tree.SetSplitStrategy(self.config.default_spl_strategy)
        while object_boundary is not None:
            object_num += 1
            object_boundary = self.NextObj()
        objects_for_train = 0
        for epoch in range(self.config.epoch):
            e = 0
            self.ResetObjLoader()
            self.tree.Clear()
            self.reference_tree.Clear()
            #construct r-tree with 1/4 datasets
            print("setup initial rtree")
            ratio_for_tree_construction = epoch % self.config.parts + 1
            for i in range(int(object_num * ratio_for_tree_construction / (self.config.parts+1))):
                object_boundary = self.NextObj()
                self.tree.DefaultInsert(object_boundary)
            fo = open('train_object.tmp', 'w')
            object_boundary = self.NextObj()

            print('filling leaf nodes')
            #fill the r-tree so every leaf is full
            objects_for_fill = 0
            cnt = 0
            while object_boundary is not None:
                if cnt % 100 == 0:
                    print(cnt)
                cnt += 1
                is_success = self.tree.TryInsert(object_boundary)
                if not is_success:
                    fo.write('{:.5f} {:.5f} {:.5f} {:.5f}\n'.format(object_boundary[0], object_boundary[1], object_boundary[2], object_boundary[3]))
                else:
                    objects_for_fill += 1
                object_boundary = self.NextObj()
            fo.close()
            #print(objects_for_fill, 'objects are used to fill leaf nodes')
            #start train
            
            cache_tree.CopyTree(self.tree.tree)
            self.reference_tree.CopyTree(cache_tree.tree)
            self.tree.CopyTree(cache_tree.tree)

            fin = open('train_object.tmp', 'r')
            period = 0
            obj_list_for_reward = []
            for line in fin:
                objects_for_train += 1
                object_boundary = [float(v) for v in line.strip().split()]
                #print('object', object_boundary)
                self.reference_tree.DefaultInsert(object_boundary)
                self.tree.DirectInsert(object_boundary)
                states, is_valid = self.tree.RetrieveSplitStates()
                triggered = False
                while states is not None:
                    triggered = True
                    states = torch.tensor(states, dtype=torch.float32)
                    with torch.no_grad():
                        q_values = self.network(states)
                        action = self.EpsilonGreedy(q_values)
                        self.tree.SplitOneStep(action)
                        steps.append((states, action, is_valid))
                        states, is_valid = self.tree.RetrieveSplitStates()

                if triggered:
                    steps.append((None, None, False))
                    split_triggered += 1

                period += 1
                obj_list_for_reward.append(object_boundary)

                if period == self.config.splits_for_update:
                    reward = self.ComputeDenseRewardForList(obj_list_for_reward)
                    #print('reward', reward)
                    debug_log.write('{}\n'.format(reward))
                    for i in range(len(steps) - 1):
                        if steps[i][2]:
                            useful_split += 1
                            self.memory.push(steps[i][0], steps[i][1], torch.tensor([reward]), steps[i+1][0])
                        else:
                            unuseful_split += 1
                    self.reference_tree.CopyTree(cache_tree.tree)
                    self.tree.CopyTree(cache_tree.tree)
                    period = 0
                    obj_list_for_reward.clear()
                    steps.clear()


                
                if period % 5 == 0:
                    l = self.Optimize()
                    loss_log.write('{}\n'.format(l))
                    if e % 500 == 0:
                        print('{} objects added, loss is {}\n'.format(e, l))
                        self.config.epsilon = max(self.config.epsilon * self.config.epsilon_decay, self.config.min_epsilon)
                    e += 1

                    if e % self.config.target_update == 0:
                        self.target_network.load_state_dict(self.network.state_dict())
            torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.epoch{}'.format(epoch)+'.mdl')
            fin.close()
        end_time = time.time()
        torch.save(self.network.state_dict(), "./model/"+self.config.model_name+'.mdl')
        print('{} objects used for training, {} cause splits\n'.format(objects_for_train, split_triggered))
        debug_log.close()
        loss_log.close()
        train_log = open('./log/train.log', 'a')
        train_log.write('{}:\n'.format(datetime.now()))
        train_log.write('{}\n'.format(self.id))
        train_log.write('{}\n'.format(self.config))
        train_log.write('training time: {}, {} objects triggered splitting, {} useful split, {} unuseful split\n'.format(end_time-start_time, split_triggered, useful_split, unuseful_split))
        train_log.write('zero reward: {}, zero reward2: {}\n'.format(reward_is_0, reward2_is_0))
        train_log.close()
        self.tree.Clear()
        self.reference_tree.Clear()
        cache_tree.Clear()



    def Train2(self):
        #first construct R-tree with 2/3 of the dataset. Train the agent with the remaining objects so that the reward should be close.
        start_time = time.time()
        loss_log = open("./log/{}.loss".format(self.config.model_name), 'w')
        debug_log = open('./reward.log', 'w')
        split_triggered = 0
        useful_split = 0
        unuseful_split = 0
        reward_is_0 = 0
        reward2_is_0 = 0
        steps = []
        object_num = 0
        self.ResetObjLoader()
        object_boundary = self.NextObj()
        while object_boundary is not None:
            object_num += 1
            object_boundary = self.NextObj()

        for epoch in range(self.config.epoch):
            e = 0
            self.ResetObjLoader()
            self.tree.Clear()
            self.reference_tree.Clear()

            for i in range(int(object_num / 3 * 2)):
                object_boundary = self.NextObj()
                self.tree.DefaultInsert(object_boundary)
            self.reference_tree.CopyTree(self.tree.tree)

            object_boundary = self.NextObj()
            trigger_period = 0
            while object_boundary is not None:

                #self.reference_tree.CopyTree(self.tree.tree)
                self.reference_tree.DefaultInsert(object_boundary)
                self.tree.DirectInsert(object_boundary)
                states, is_valid = self.tree.RetrieveSplitStates()
                triggered = False
                while states is not None:
                    triggered = True
                    states = torch.tensor(states, dtype=torch.float32)
                    with torch.no_grad():
                        q_values = self.network(states)
                    action = self.EpsilonGreedy(q_values)
                    self.tree.SplitOneStep(action)
                    steps.append((states, action, is_valid))
                    states, is_valid = self.tree.RetrieveSplitStates()

                if triggered:
                    split_triggered += 1
                    trigger_period += 1
                    #print('triggered', trigger_period, split_triggered)
                    steps.append((None, None, False))

                if trigger_period == self.config.splits_for_update:
                    reward = self.ComputeReward()
                    debug_log.write('{}\n'.format(reward))
                    for i in range(len(steps) - 1):
                        if steps[i][2]:
                            useful_split += 1
                            self.memory.push(steps[i][0], steps[i][1], torch.tensor([reward]), steps[i+1][0])
                        else:
                            unuseful_split += 1
                    self.reference_tree.CopyTree(self.tree.tree)
                    trigger_period = 0


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
        train_log.write('{}\n'.format(self.id))
        train_log.write('{}\n'.format(self.config))
        train_log.write('training time: {}, {} objects triggered splitting, {} useful split, {} unuseful split\n'.format(end_time-start_time, split_triggered, useful_split, unuseful_split))
        train_log.write('zero reward: {}, zero reward2: {}\n'.format(reward_is_0, reward2_is_0))
        train_log.close()
        loss_log.close()
        debug_log.close()

    def Train(self):
        start_time = time.time()
        loss_log = open("./log/{}.loss".format(self.config.model_name), 'w')
        debug_log = open('./reward.log', 'w')
        split_triggered = 0
        useful_split = 0
        unuseful_split = 0
        reward_is_0 = 0
        reward2_is_0 = 0
        steps = []
        for epoch in range(self.config.epoch):
            e = 0
            self.ResetObjLoader()
            self.tree.Clear()
            self.reference_tree.Clear()
            object_boundary = self.NextObj()
            trigger_period = 0
            while object_boundary is not None:

                #self.reference_tree.CopyTree(self.tree.tree)
                self.reference_tree.DefaultInsert(object_boundary)
                self.tree.DirectInsert(object_boundary)
                states, is_valid = self.tree.RetrieveSplitStates()
                triggered = False
                while states is not None:
                    triggered = True
                    states = torch.tensor(states, dtype=torch.float32)
                    with torch.no_grad():
                        q_values = self.network(states)
                    action = self.EpsilonGreedy(q_values)
                    self.tree.SplitOneStep(action)
                    steps.append((states, action, is_valid))
                    states, is_valid = self.tree.RetrieveSplitStates()

                if triggered:
                    split_triggered += 1
                    trigger_period += 1
                    #print('triggered', trigger_period, split_triggered)
                    steps.append((None, None, False))

                if trigger_period == self.config.splits_for_update:
                    reward = self.ComputeReward()
                    debug_log.write('{}\n'.format(reward))
                    for i in range(len(steps) - 1):
                        if steps[i][2]:
                            useful_split += 1
                            self.memory.push(steps[i][0], steps[i][1], torch.tensor([reward]), steps[i+1][0])
                        else:
                            unuseful_split += 1
                    self.reference_tree.CopyTree(self.tree.tree)
                    trigger_period = 0


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
        loss_log.close()
        debug_log.close()



if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cpu")

    if args.action == 'train':
        spl_learner = SplitLearner()
        spl_learner.Initialize(args)
        spl_learner.Train5()
    if args.action == 'test':
        spl_learner = SplitLearner()
        spl_learner.Initialize(args)
        spl_learner.Test2()
    if args.action == 'test10':
        spl_learner = SplitLearner()
        spl_learner.Initialize(args)
        spl_learner.Test10_2()

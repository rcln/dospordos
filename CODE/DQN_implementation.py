# -*- coding: utf-8 -*-
# import h5py
import os
import random
import sys
import logging

import numpy as np
import agent
import preprocessing as prep
import pickle
from random import randint

from Baselines import Baselines
from Evaluation import Evaluation
from environment import Environment
from agent import Agent, Network
from Sars import Sars
from sklearn.externals import joblib

path_history = "../DATA/history"
path_model = "../DATA/dqn/model_nn.h5"

load_model = False
# can we have repeatables, ask Pegah ta daaaa : PA response: no it should not be repeatable samples


class DQN:
    """we have two DQN approaches in general: 1- DQN + normal NE 2 - DQN + normal NE + regular expresions"""

    def __init__(self, env_, agent_, list_users_, is_RE, logger, name):
        """
        :param env_:
        :param agent_:
        :param list_users_:
        :param is_RE: verifies if we use regular expressions in name entity extraction or not.
        """

        self.env = env_
        self.agent = agent_
        self.is_RE = is_RE
        self.logger = logger
        self.name = name
        self.path_replay_memory = '/../DATA/' + self.name + '_replay_memory.pkl'
        # Desc: loading users
        self.list_users = list_users_

        self.reward_matrix = []
        self.accuracy_matrix = []
        self.measure_results_matrix = []
        self.base_ma_list = []
        self.base_ctg_list = []

        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

        # self.callbacks = [agent.EarlyStopByLossVal()]
        # ToDo: Note to Pegah, another callback with it can be stopped if there's no improvement with a min_delta .
        # after some epochs
        self.callbacks = [agent.EarlyStopByLossVal(value=0.01),
                          agent.EarlyStopping(patience=3)]

    def get_random_elements(self, ar: list, number):
        """
        choose number of non repetitive elements from ar list.
        :param ar:
        :param number:
        :return:
        """
        output = random.sample(ar, number)
        element_list = []
        for i in output:  # range(0, number):
            # element_list.append(ar[np.random.randint(0, len(ar))])
            element_list.append(i)
        return element_list

    def interpret_action(self, action_vector):
        num_l = np.nonzero(action_vector)
        num = num_l[1][0]

        actions_db = ("delete current_db", "add current_db", "keep current_db")
        actions_grid = ("next_snippet", "change_queue")

        print("Actions:: ", actions_grid[int(num / 3)], ";;", actions_db[num % 3])

    def bad_franky(self, ar):
        return np.isnan(ar).any() or np.isinf(ar).any()

    def training_phase(self, minibatch):

        if os.path.exists(self.env.path_weights):
            self.agent.network.load_weights(self.env.path_weights)
        # Desc: agent.print_model()
        ###shuffle(self.list_users)
        len_list_user = len(self.list_users)

        if os.path.exists(os.getcwd() + self.path_replay_memory):
            replay_memory = joblib.load(os.getcwd() + self.path_replay_memory)
        else:  # Desc: generate first replay memory
            print("Getting replay memory")
            replay_memory_ar = []
            # TODO PA: maybe 999 replay training is not enough because we have 4518 users in total
            while len(replay_memory_ar) <= minibatch:
                random_user = self.list_users[randint(0, len_list_user)]
                s, pass_us = self.env.reset(random_user, is_RE=self.is_RE)

                if pass_us:
                    continue
                for x in range(0, 100):  # TODO PA: why 30 times?. Answer it was for debugging purpose
                    a = Sars.get_random_action_vector_pa(6)
                    r, s_prime, done = self.env.step_pa(self.agent.actions_to_take_pa(a), a, is_RE=self.is_RE)
                    replay_memory_ar.append(Sars(s, a, r, s_prime, False))
                    s = s_prime
                print('gold standard', self.env.current_name, self.env.golden_standard_db)
                print('extracted name entities', self.env.university_name_pa, self.env.date_pa)
            print("Saving replay memory")
            joblib.dump(replay_memory_ar, os.getcwd() + self.path_replay_memory)
            print("Saved replay memory")
            replay_memory = replay_memory_ar

        return replay_memory

    def generate_action(self, i, length):
        action_vector = [0] * length
        action_vector[i] = 1
        action_vector = np.array([action_vector])
        return action_vector

    def get_max_action(self, state, network):
        arg_max = []
        for i in range(6):
            action_vector = self.generate_action(i, 6)
            in_vector = np.concatenate([state, action_vector], axis=1)
            arg_max.append(network.predict(in_vector))
        if self.bad_franky(arg_max):
            print("The project is in danger :(, out_vector ", arg_max)
        action_vector = [0] * 6
        action_vector[arg_max.index(max(arg_max))] = 1
        return action_vector

    def replay_memory(self, size):
        # Desc: loading replayed memory
        if os.path.exists(os.getcwd() + self.path_replay_memory):
            replay_memory = joblib.load(os.getcwd() + self.path_replay_memory)
        else:
            # Desc: generate first replay memory
            # MDP framework for Information Extraction (Training Phase)
            replay_memory = self.training_phase(size)
        return replay_memory

    def get_action_with_probability(self, state, networkQN, eps):

        p = np.random.random()

        if p < eps:
            action_vector = np.zeros([1, 6])
            action_vector[0, np.random.randint(0, 6)] = 1
            action_vector = action_vector[0]
            # action_vector = Sars.get_random_action_vector(6)[0]
        else:
            # Desc: a = argmaxQ(s,a)
            action_vector = self.get_max_action(state, networkQN)

        return action_vector

    def refill_memory(self, replay_memory, state, action_vector, reward, next_state, size):

        if len(replay_memory) < size:
            replay_memory.append(Sars(state, action_vector, reward, next_state))
        else:
            print("ADDING TO MEMORY...", " reward type", type(reward), " reward", reward)
            del replay_memory[0]
            replay_memory.append(Sars(state, action_vector, reward, next_state))

        return replay_memory

    def DoubleDQN(self, gamma, eps, training_replay_size):

        # TODO: this function should be double checked.

        "two networks are required"
        mainQN = self.agent.network  # Q for finding the max action: for all s, argmax_a mainQN(s, a)
        targetQN = Network((28,))  # Q for evaluating actions: for all s, a, targetQN(s, a)
        targetQN.model.set_weights(mainQN.model.get_weights())

        # Desc: loading replayed memory
        replay_memory = self.replay_memory(training_replay_size)

        # Epochs
        e_count = 0
        stop_train = False

        with open('tmp_record', 'w') as f:
            f.write("0")
        for us in self.list_users: #[35:36]:

            with open('tmp_record', 'r') as f:
                if f.readline() == "1":
                    stop_train = True
            if stop_train:
                print("Train has stopped due to callback EarlyStopByLoss")
                return

            # initial state
            state, err = self.env.reset(us, is_RE=self.is_RE)

            if err:
                continue

            done = False
            # Double DQN
            counter = 0
            # episodes
            tmp_reward = 0
            # An epoch is an iteration here
            while not done:

                if counter > 1000:
                    print('we use break option')
                    break

                """Select an action with an epsilon probability"""
                action_vector = self.get_action_with_probability(state, mainQN, eps)
                # action_vector = self.get_action_with_probability(state, self.agent.network, eps)

                # Observe reward and new state
                reward, next_state, done = self.env.step_pa(self.agent.actions_to_take_pa(action_vector), action_vector,
                                                            is_RE=self.is_RE)

                self.logger.info('Reward:: ' + str(reward) + "," + str(counter) + "," + str(e_count))
                tmp_reward = tmp_reward + reward

                replay_memory = self.refill_memory(replay_memory, state, action_vector, reward, next_state, 1000)

                # Desc: reward + gamma * Q(s', argmax_a'(Q[s',a', w_main]), w_target ) - Q[s,a, w_main])
                # this part is for learning the Q function using gradient descent
                X_train = []
                Y_train = []

                tempo = self.get_random_elements(replay_memory, 1000)

                for sample in tempo:

                    # if state is terminal
                    if self.env._check_grid() or (
                            sample.s_prime[0, 15] == sample.s_prime[0, 16] == sample.s_prime[0, 17]):
                        t = np.array([sample.r])
                        print(" GETTING REWARD JUST FROM SAMPLE.R  t=", t, "t. shape")
                    else:
                        action_vector = self.get_max_action(sample.s, mainQN)
                        t_vector = np.concatenate((sample.s_prime, np.array([action_vector])), axis=1)
                        t = sample.r + gamma * targetQN.predict(t_vector)[0][0]

                    if not type(sample.a) is list:
                        tempo = (sample.a).tolist()
                    else:
                        tempo = sample.a

                    x_train = (sample.s[0]).tolist() + tempo

                    if len(X_train) == 0:
                        X_train = np.asarray([x_train])
                    else:
                        X_train = np.append(np.asarray(X_train), np.asarray([x_train]), axis=0)

                    # Q(s,a) computed using Q-learning method
                    Y_train.append(t)

                Y_train = np.array(Y_train)

                targetQN.model.set_weights(mainQN.model.get_weights())

                mainQN.fit(X_train, Y_train, 1, len(X_train), callbacks=self.callbacks)
                mainQN.save_weights(self.env.path_weights)

                state = next_state
                counter += 1

            # TODO PA: do the wieghts change during the learning process in the NN automatically?
            self.agent.network.save_weights(self.env.path_weights)

            """get entities by following the optimal policy"""
            # print("Counter", self.get_best_entities_with_optimal_policy(eps, us))
            print('Gold standards', self.env.current_name, self.env.golden_standard_db)
            print('Extracted entities', self.env.university_name_pa, self.env.date_pa)

            e_count = e_count + 1
        return

    def deep_QN(self, gamma, eps, training_replay_size):
        # TODO PA: the gamma should be increased to 0.95 or 0.99 after the first test
        # TODO PA: the eps should be tested with the small values
        # Desc: loading users
        # Desc: loading replayed memory
        replay_memory = self.replay_memory(training_replay_size)
        # Epochs
        e_count = 0
        stop_train = False
        # train episodes
        with open('tmp_record', 'w') as f:
            f.write("0")
        for us in self.list_users: #[35:36]:
            with open('tmp_record', 'r') as f:
                if f.readline() == "1":
                    stop_train = True
            if stop_train:
                print("Train has stopped due to callback EarlyStopByLoss")
                return
            # reset episode with new user and get initial state
            state, err = self.env.reset(us, is_RE=self.is_RE)
            if err:
                continue
            done = False
            # DQN with experience replace
            counter = 0
            # epoch
            tmp_reward = 0
            while not done:
                if counter > 1000:
                    print('we use break option')
                    break

                """Select an action with an epsilon probability"""
                action_vector = self.get_action_with_probability(state, self.agent.network, eps)

                # Observe reward and new state
                reward, next_state, done = self.env.step_pa(self.agent.actions_to_take_pa(action_vector), action_vector,
                                                            is_RE=self.is_RE)
                self.logger.info('Reward:: ' + str(reward) + ", Step:: " + str(counter) + ", Episode:: " + str(e_count))
                tmp_reward = tmp_reward + reward

                replay_memory = self.refill_memory(replay_memory, state, action_vector, reward, next_state, 1000)

                # Desc: Q[s,a] = Q[s,a] + learning_rate*(reward + discount* max_a'(Q[s',a']) - Q[s,a])
                # this part is for learning the Q function using gradient descent
                X_train = []
                Y_train = []

                tempo = self.get_random_elements(replay_memory, 1000)

                for sample in tempo:

                    # if state is terminal
                    if self.env._check_grid() or (
                            sample.s_prime[0, 15] == sample.s_prime[0, 16] == sample.s_prime[0, 17]):
                        t = np.array([sample.r])
                        print(" GETTING REWARD JUST FROM SAMPLE.R  t=", t, "t. shape")
                    else:
                        target_ar = []
                        for i in range(6):
                            action_vector = self.generate_action(i, 6)
                            t_vector = np.concatenate((sample.s_prime, action_vector), axis=1)
                            target_ar.append(self.agent.network.predict(t_vector))

                        if self.bad_franky(target_ar):
                            print("Target_ar that is in training is bad, target_ar", target_ar)

                        t = sample.r + gamma * max(target_ar)[0][0]

                    if not type(sample.a) is list:
                        tempo = (sample.a).tolist()
                    else:
                        tempo = sample.a

                    x_train = (sample.s[0]).tolist() + tempo

                    if len(X_train) == 0:
                        X_train = np.asarray([x_train])
                    else:
                        X_train = np.append(np.asarray(X_train), np.asarray([x_train]), axis=0)

                    # Q(s,a) computed using Q-learning method
                    Y_train.append(t)
                    # Q(s,a) computed using the neural network

                Y_train = np.array(Y_train)

                self.agent.network.fit(X_train, Y_train, 1, len(X_train), callbacks=self.callbacks)
                state = next_state

                counter += 1

            # TODO PA: does the wieghts change during the learning process in the NN automatically?
            self.agent.network.save_weights(self.env.path_weights)

            """get entities by following the optimal policy"""
            print('Gold standards', self.env.current_name, self.env.golden_standard_db)
            print('Extracted entities', self.env.university_name_pa, self.env.date_pa)

            e_count = e_count + 1

        return

    def testing(self, eps):
        if os.path.exists(self.env.path_weights):
            self.agent.network.load_weights(self.env.path_weights)
            print('LOADING WEIGHTS!')
        for us in self.list_users:
            self.get_best_entities_with_optimal_policy(eps=eps, us=us)
        pickle.dump(self.measure_results_matrix, open('../DATA/'+self.name+'_mrm.pkl', 'wb'))
        pickle.dump(self.reward_matrix, open('../DATA/'+self.name+'_rm.pkl', 'wb'))
        pickle.dump(self.base_ctg_list, open('../DATA/'+self.name+'_ctg.pkl', 'wb'))
        pickle.dump(self.base_ma_list, open('../DATA/'+self.name+'_ma.pkl', 'wb'))
        pickle.dump(self.accuracy_matrix, open('../DATA/'+self.name+'_acc.pkl', 'wb'))

    def get_best_entities_with_optimal_policy(self, eps, us):
        reward_list = [0]
        measure_results_list = []

        # initial state
        state, err = self.env.reset(us, is_RE=self.is_RE)

        if err:
            return -1

        done = False
        counter = 0
        base = Baselines(self.env, self.agent, [], is_RE=self.is_RE)

        # epoch
        # if you want to observe reward accumulation or accuracy for the test set, it should be in each itetation of the following loop.
        while not done:
            # for i in range(50):

            if counter > 1000:
                print('we use break option')
                return counter

            """Select an action with an epsilon probability"""
            action_vector = self.get_action_with_probability(state, self.agent.network, eps)
            # Observe reward and new state
            reward, next_state, done = self.env.step_pa(self.agent.actions_to_take_pa(action_vector), action_vector,
                                                        is_RE=self.is_RE)

            reward_list.append((reward+reward_list[-1]))
            state = next_state
            self.logger.info(' Test.. Reward:: ' + str(reward) + ", step::" + str(counter) + ", user::" + str(us))
            counter += 1
            eval = Evaluation(self.env.golden_standard_db, self.env.university_name_pa, self.env.date_pa)
            self.accuracy_matrix.append(eval.total_accuracy())

        measuring_results = eval.get_measuring_results()
        measure_results_list.append(measuring_results)

        self.reward_matrix.append(reward_list[1:])
        self.measure_results_matrix.append(measure_results_list)
        entities, gold = base.baseline_agregate_NE(us)

        self.base_ma_list.append(base.majority_aggregation(entities, gold))
        self.base_ctg_list.append(base.closest_to_gold(entities, gold))
        print("Done with user: ", str(us))
        return counter


if __name__ == "__main__":
    # ToDo Pegah: Please check the file training_script.py to run DQN
    pass

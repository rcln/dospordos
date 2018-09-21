# -*- coding: utf-8 -*-
import h5py
import os
import random
import sys

import numpy as np
import preprocessing as prep
import pickle
from random import shuffle, randint

from Evaluation import Evaluation
from environment import Environment
from agent import Agent, Network
from Sars import Sars
from sklearn.externals import joblib

path_replay_memory = "/../DATA/replay_memory.pkl"
path_history = "../DATA/history"
path_model = "../DATA/dqn/model_nn.h5"

load_model = False


# can we have repeatables, ask Pegah ta daaaa : PA response: no it should not be repeatable samples


class DQN:
    """we have two DQN approaches in general: 1- DQN + normal NE 2 - DQN + normal NE + regular expresions"""

    def __init__(self, env_, agent_, list_users_, is_RE):
        """

        :param env_:
        :param agent_:
        :param list_users_:
        :param is_RE: verifies if we use regular expressions in name entity extraction or not.
        """

        self.env = env_
        self.agent = agent_
        self.is_RE = is_RE

        # Desc: loading users
        self.list_users = list_users_

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
        shuffle(self.list_users)
        len_list_user = len(self.list_users)

        if os.path.exists(os.getcwd() + path_replay_memory):
            replay_memory = joblib.load(os.getcwd() + path_replay_memory)
        else:  # Desc: generate first replay memory
            print("Getting replay memory")
            replay_memory_ar = []
            # TODO PA: maybe 999 replay training is not enough because we have 4518 users in total
            while len(replay_memory_ar) <= minibatch:

                random_user = self.list_users[randint(0, len_list_user)]
                s, pass_us = self.env.reset(random_user, is_RE=self.is_RE)

                if pass_us:
                    continue

                for x in range(0, 100):  # TODO PA: why 30 times?
                    a = Sars.get_random_action_vector_pa(6)
                    r, s_prime, done = self.env.step_pa(self.agent.actions_to_take_pa(a), a, is_RE=self.is_RE)
                    replay_memory_ar.append(Sars(s, a, r, s_prime, False))
                    s = s_prime

                print('gold standars', self.env.current_name, self.env.golden_standard_db)
                print('extracted name entities', self.env.university_name_pa, self.env.date_pa)

            print("Saving replay memory")
            joblib.dump(replay_memory_ar, os.getcwd() + path_replay_memory)
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
        if os.path.exists(os.getcwd() + path_replay_memory):
            replay_memory = joblib.load(os.getcwd() + path_replay_memory)
        else:
            # Desc: generate first replay memory
            # MDP framework for Information Extraction (Traiing Phase)
            replay_memory = self.training_phase(size)

        return replay_memory

    def get_action_with_probability(self, state, networkQN, eps):

        p = np.random.random()

        if p < eps:
            action_vector = Sars.get_random_action_vector(6)[0]
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

        for us in self.list_users[35:36]:
            # initial state
            state, err = self.env.reset(us, is_RE=self.is_RE)

            if err:
                continue

            done = False

            # Double DQN
            counter = 0

            # episodes
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

                # Todo Ask Pegah about replay memory, ask her opinion...?
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
                mainQN.fit(X_train, Y_train, 1, len(X_train))
                # mainQN.save_weights(self.env.path_weights)

                state = next_state
                counter += 1

            # TODO PA: do the wieghts change during the learning process in the NN automatically?
            self.agent.network.save_weights(self.env.path_weights)

            """get entities by following the optimal policy"""
            print("Counter", self.get_best_entities_with_optimal_policy(us))
            print('Gold standards', self.env.current_name, self.env.golden_standard_db)
            print('Extracted entities', self.env.university_name_pa, self.env.date_pa)

            eval = Evaluation(self.env.golden_standard_db, self.env.university_name_pa, self.env.date_pa)
            print(eval.total_accuray())

        return

    def deep_QN(self, gamma, eps, training_replay_size):
        # TODO PA: the gamma should be increased to 0.95 or 0.99 after the first test
        # TODO PA: the eps should be tested with the small values

        history = []

        # Desc: loading users
        # list_users = sorted(list(map(int, os.listdir(self.env.path))))

        # Desc: loading replayed memory
        replay_memory = self.replay_memory(training_replay_size)

        # train episodes
        for us in self.list_users[35:36]:

            # reset episode with new user and get initial state
            state, err = self.env.reset(us, is_RE=self.is_RE)
            episode = {}
            if err:
                continue
            episode[len(history)] = []

            done = False

            # DQN with experience replace
            counter = 0
            while not done:
                # for i in range(50):

                if counter > 1000:
                    print('we use break option')
                    break

                """Select an action with an epsilon probability"""
                action_vector = self.get_action_with_probability(state, self.agent.network, eps)

                # Observe reward and new state
                reward, next_state, done = self.env.step_pa(self.agent.actions_to_take_pa(action_vector), action_vector,
                                                            is_RE=self.is_RE)

                episode[len(history)].append(reward)

                # Todo Ask Pegah about replay memory ask her opinion...?
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

                self.agent.network.fit(X_train, Y_train, 1, len(X_train))
                state = next_state

                counter += 1

            # TODO PA: does the wieghts change during the learning process in the NN automatically?
            self.agent.network.save_weights(self.env.path_weights)
            history.append(episode)
            print("HISTORY..", history, "type.", type(history))

            pickle.dump(history, open(path_history, 'wb'))

            """get entities by following the optimal policy"""
            print("Counter", self.get_best_entities_with_optimal_policy(us))
            print('Gold standards', self.env.current_name, self.env.golden_standard_db)
            print('Extracted entities', self.env.university_name_pa, self.env.date_pa)

            eval = Evaluation(self.env.golden_standard_db, self.env.university_name_pa, self.env.date_pa)
            print(eval.total_accuray())

        return

    def get_best_entities_with_optimal_policy(self, us):

        # initial state
        state, err = self.env.reset(us, is_RE=self.is_RE)

        done = False
        counter = 0

        while not done:
            # for i in range(50):

            if counter > 1000:
                print('we use break option')
                return counter

            action_vector = self.get_max_action(state, self.agent.network)
            # Observe reward and new state
            reward, next_state, done = self.env.step_pa(self.agent.actions_to_take_pa(action_vector), action_vector,
                                                        is_RE=self.is_RE)
            state = next_state

            counter += 1

        return counter


if __name__ == "__main__":

    # TODO TP needd to change this env to reflect what datasource we want to use
    env = Environment()

    if not os.path.exists(env.path_count_vect) or not os.path.exists(env.path_tfidf_vect):
        print("Training BOW vectors")
        prep.file_to_pickle_vectorizer(os.getcwd() + "/../DATA/")
        print("---FIT COMPLETED----")

    # TODO get the second number automatically ... env.tf_vectorizer.shape[0]
    # Desc: agent = Agent(env, (27 + 27386,))
    # len_vect = env.tf_vectorizer.transform([""]).toarray()
    # print(len(len_vect))

    agent = Agent(env, (28,))  # + len_vect.shape[1],)) # the comma is very important
    list_users = sorted(list(map(int, os.listdir(env.path))))

    dqn = DQN(env, agent, list_users, is_RE=True)

    try:
        dqn.deep_QN(gamma= 0.95, eps = 0.1, training_replay_size=2000)
        # dqn.DoubleDQN(gamma= 0.95, eps = 0.1, training_replay_size= 2000)

    except KeyboardInterrupt:
        print("\n\n-----Interruption-----\nSaving weights")
        agent.network.save_weights(env.path_weights)
        print("Weights saved")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

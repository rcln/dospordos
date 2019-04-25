# -*- coding: utf-8 -*-
# import h5py
import os
import sys
import logging

import numpy as np
import agent
import pickle
from random import randint

from Baselines import Baselines
from Evaluation import Evaluation
from agent import Agent, Network
from Sars import Sars
from sklearn.externals import joblib
from DQN import DQN
from keras.callbacks import History

history = History()

load_model = False


# can we have repeatables, ask Pegah ta daaaa : PA response: no it should not be repeatable samples


class DQN_NN:
    """we have two DQN approaches in general: 1- DQN + normal NE 2 - DQN + normal NE + regular expresions"""

    def __init__(self, env_, agent_, list_users_, is_RE, logger, name):
        """
        :param env_:
        :param agent_:
        :param list_users_:
        :param is_RE: verifies if we use regular expressions in name entity extraction or not.
        """

        dqnn = DQN(env_, agent_, list_users_, is_RE, logger, name)

        self.dqnn = dqnn

        self.env = dqnn.env
        self.agent = dqnn.agent
        self.is_RE = dqnn.is_RE
        self.logger = dqnn.logger
        self.name = dqnn.name
        self.path_replay_memory = dqnn.path_replay_memory
        # Desc: loading users
        self.list_users = dqnn.list_users

        self.reward_matrix = dqnn.reward_matrix
        self.accuracy_matrix = dqnn.accuracy_matrix
        self.measure_results_matrix = dqnn.measure_results_matrix

        "new measures added by Pegah"
        self.used_users = dqnn.used_users
        self.final_queries = dqnn.final_queries
        self.num_changed_queries = dqnn.num_changed_queries
        self.percentage_used_snippets = dqnn.percentage_used_snippets
        self.trajectories_results = dqnn.trajectories_results
        self.gold_standards = dqnn.gold_standards

        self.base_ma_list = dqnn.base_ma_list
        self.base_ctg_list = dqnn.base_ctg_list

        self.action_size = dqnn.action_size

        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

        # after some epochs
        self.callbacks = [agent.EarlyStopByLossVal(value=0.01),
                          agent.EarlyStopping(patience=3)]

    def training_phase(self, minibatch):

        if os.path.exists(self.env.path_weights):
            self.agent.network.load_weights(self.env.path_weights)
        # Desc: agent.print_model()
        ###shuffle(self.list_users)
        len_list_user = len(self.list_users)-1

        if os.path.exists(os.getcwd() + self.path_replay_memory):
            replay_memory = joblib.load(os.getcwd() + self.path_replay_memory)
        else:  # Desc: generate first replay memory
            print("Getting replay memory")
            replay_memory_ar = []
            # TODO PA: maybe 999 replay training is not enough because we have 4518 users in total
            while len(replay_memory_ar) <= minibatch:
                tempo = randint(0, len_list_user)
                random_user = self.list_users[tempo]
                s, pass_us = self.env.reset(random_user, is_RE=self.is_RE)

                if pass_us:
                    continue
                for x in range(0, 100):  # TODO PA: why 100 times?
                    a = Sars.get_random_action_vector_pa(self.action_size)
                    r, s_prime, done = self.env.step_pa(self.agent.actions_to_take_pa(a), a, is_RE=self.is_RE)
                    replay_memory_ar.append(Sars(s, a, r, s_prime, False))
                    s = s_prime

            print("Saving replay memory")
            joblib.dump(replay_memory_ar, os.getcwd() + self.path_replay_memory)
            print("Saved replay memory")
            replay_memory = replay_memory_ar

        return replay_memory

    def get_max_action(self, state, network):
        arg_max = []
        for i in range(self.action_size):
            action_vector = self.dqnn.generate_action(i, self.action_size)
            in_vector = np.concatenate([np.array(state), np.array(action_vector)], axis=1)
            arg_max.append(network.predict(in_vector))
        if self.dqnn.bad_franky(arg_max):
            print("The project is in danger :(, out_vector ", arg_max)
        action_vector = [0] * self.action_size
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
            action_vector = np.zeros([1, self.action_size])
            action_vector[0, np.random.randint(0, self.action_size)] = 1
            action_vector = action_vector[0]
            # action_vector = Sars.get_random_action_vector(self.action_size)[0]
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

    def DoubleDQN(self, gamma, eps, training_replay_size, is_db_v2):

        # TODO: this function should be double checked.

        "two networks are required"
        mainQN = self.agent.network  # Q for finding the max action: for all s, argmax_a mainQN(s, a)
        if is_db_v2: #Q for evaluating actions: for all s, a, targetQN(s, a)
            targetQN = Network((25,))
        else:
            targetQN = Network((24,))

        targetQN.model.set_weights(mainQN.model.get_weights())

        # Desc: loading replayed memory
        replay_memory = self.replay_memory(training_replay_size)

        # Epochs
        e_count = 0
        stop_train = False

        with open('tmp_record', 'w') as f:
            f.write("0")
        for us in self.list_users:  # [35:36]:

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

                tempo = self.dqnn.get_random_elements(replay_memory, 100)

                for sample in tempo:

                    # if state is terminal
                    if self.env.env_core._check_grid():
                            #or (
                            #sample.s_prime[0, 15] == sample.s_prime[0, 16] == sample.s_prime[0, 17]):
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

                mainQN.fit(X_train, Y_train, 10, len(X_train), callbacks=self.callbacks)
                mainQN.save_weights(self.env.path_weights)

                state = next_state
                counter += 1

            # TODO PA: do the weights change during the learning process in the NN automatically?
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
        user_counter = 0
        stop_train = False
        # train episodes
        with open('tmp_record', 'w') as f:
            f.write("0")
        for us in self.list_users:
            user_counter += 1
            # with open('tmp_record', 'r') as f:
            #     if f.readline() == "1":
            #         stop_train = True
            # if stop_train:
            #     print("Train has stopped due to callback EarlyStopByLoss")
            #     return

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

                #Select an action with an epsilon probability
                action_vector = self.get_action_with_probability(state, self.agent.network, eps)

                # Observe reward and new state
                reward, next_state, done = self.env.step_pa(self.agent.actions_to_take_pa(action_vector), action_vector,
                                                            is_RE=self.is_RE)
                self.logger.info('Reward:: ' + str(reward) + ", Step:: " + str(counter) + ", Episode:: " + str(e_count) +
                                 ", action::" + str(action_vector))
                tmp_reward = tmp_reward + reward

                replay_memory = self.refill_memory(replay_memory, state, action_vector, reward, next_state, 1000)

                # Desc: Q[s,a] = Q[s,a] + learning_rate*(reward + discount* max_a'(Q[s',a']) - Q[s,a])
                # this part is for learning the Q function using gradient descent
                X_train = []
                Y_train = []

                tempo = self.dqnn.get_random_elements(replay_memory, 100)

                for sample in tempo:
                    # if state is terminal
                    if self.env.env_core._check_grid():
                        # or (
                        #    sample.s_prime[0, 15] == sample.s_prime[0, 16] == sample.s_prime[0, 17]):
                        t = np.array([sample.r])
                        print(" GETTING REWARD JUST FROM SAMPLE.R  t=", t, "t. shape")
                    else:
                        target_ar = []
                        for i in range(self.action_size):
                            action_vector = self.dqnn.generate_action(i, self.action_size)

                            t_vector = np.concatenate((sample.s_prime, np.array(action_vector)), axis=1)
                            target_ar.append(self.agent.network.predict(t_vector))

                        if self.dqnn.bad_franky(target_ar):
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
                hist = self.agent.network.fit(X_train, Y_train, 10, len(X_train), callbacks=self.callbacks)

                state = next_state

                counter += 1

                # TODO PA: does the weights change during the learning process in the NN automatically?
                self.agent.network.save_weights(self.env.path_weights)


            #get entities by following the optimal policy
            #print('Gold standards', self.env.env_core.current_name, self.env.env_core.golden_standard_db)
            #print('Extracted entities', self.env.env_core.university_name_pa, self.env.env_core.date_pa)

            e_count = e_count + 1
            if user_counter%10 == 0:
                self.agent.network.save_weights('../DATA/weights_' + str(hist['loss'][-1]) + '_' + str(user_counter) + '.h5')
        return

    def testing(self, eps, iteration_test):
        for k in range(iteration_test):
            if os.path.exists(self.env.path_weights):
                print(self.env.path_weights)
                self.agent.network.load_weights(self.env.path_weights)
                print('LOADING WEIGHTS!')
            for us in self.list_users:
                print('user', us)
                self.get_best_entities_with_optimal_policy(eps=eps, us=us)

            "each element of this vector represents Pu, Ru, Fu, Py, Ry, Fy respectively"
            pickle.dump(self.measure_results_matrix, open('../DATA/' + self.name + '_mrm_' + str(k) + '.pkl', 'wb'))
            pickle.dump(self.reward_matrix, open('../DATA/' + self.name + '_rm_' + str(k) + '.pkl', 'wb'))
            pickle.dump(self.base_ctg_list, open('../DATA/' + self.name + '_ctg_' + str(k) + '.pkl', 'wb'))
            pickle.dump(self.base_ma_list, open('../DATA/' + self.name + '_ma_' + str(k) + '.pkl', 'wb'))
            pickle.dump(self.accuracy_matrix, open('../DATA/' + self.name + '_acc_' + str(k) + '.pkl', 'wb'))

            "new added parameters by Pegah"
            pickle.dump(self.used_users, open('../DATA/' + self.name + '_uu_' + str(k) + '.pkl', 'wb'))
            pickle.dump(self.final_queries, open('../DATA/' + self.name + '_queries_' + str(k) + '.pkl', 'wb'))
            pickle.dump(self.num_changed_queries, open('../DATA/' + self.name + '_nc_queries_' + str(k) + '.pkl', 'wb'))
            pickle.dump(self.percentage_used_snippets, open('../DATA/' + self.name + '_per_snippets_' + str(k) + '.pkl', 'wb'))
            pickle.dump(self.trajectories_results, open('../DATA/' + self.name + '_trajectories_' + str(k) + '.pkl', 'wb'))
            pickle.dump(self.gold_standards, open('../DATA/' + self.name + '_gold_standards_' + str(k) + '.pkl', 'wb'))

        pass

    def get_best_entities_with_optimal_policy(self, eps, us):
        reward_list = [0]
        accuracy_list = []
        measure_results_list = []

        # initial state
        state, err = self.env.reset(us, is_RE=self.is_RE)

        if err:
            return -1

        done = False
        counter = 0
        count_change_query = 0
        base = Baselines(self.env, self.agent, [], is_RE=self.is_RE)

        queries_total = [self.env.env_core.queues[k].qsize() for k in self.env.env_core.queues.keys()]

        # epoch
        # if you want to observe reward accumulation or accuracy for the test set, it should be in each itetation of the following loop.
        while not done:
        #for i in range(10):

            if counter > 1000:
                print('we use break option')
                return counter

            """Select an action with an epsilon probability"""
            action_vector = self.get_action_with_probability(state, self.agent.network, eps)
            # Observe reward and new state
            reward, next_state, done = self.env.step_pa(self.agent.actions_to_take_pa(action_vector), action_vector,
                                                        is_RE=self.is_RE)

            #print([self.env.env_core.queues[k].qsize() for k in self.env.env_core.queues.keys()])
            reward_list.append(reward + reward_list[-1])
            state = next_state

            if action_vector[-1] == 1:
                count_change_query += 1

            queries = [self.env.env_core.queues[k].qsize() for k in self.env.env_core.queues.keys()]
            #self.logger.info(  ' Test.. Reward:: ' + str(reward) + ", step::" + str(counter) + ", user::" + str(us)+
            #                   ", action::" + str(action_vector) + ", action activation::" + str(action_vector[-1])+
            #    ", Queries::" + str(queries))
            counter += 1

            eval = Evaluation(self.env.env_core.golden_standard_db, self.env.env_core.university_name_pa, self.env.env_core.date_pa)
            self.accuracy_matrix.append(eval.total_accuracy())
            accuracy_list.append(eval.total_accuracy())

        tot_result = self.env.env_core.university_name_pa, self.env.env_core.date_pa
        #total_prec = eval.total_accuracy()

        self.accuracy_matrix.append(accuracy_list)

        measuring_results = eval.get_measuring_results()
        measure_results_list.append(measuring_results)

        self.reward_matrix.append(reward_list)
        self.measure_results_matrix.append(measure_results_list)
        entities, gold = base.baseline_agregate_NE(us)

        self.base_ma_list.append(base.majority_aggregation(entities, gold))
        self.base_ctg_list.append(base.closest_to_gold(entities, gold))

        self.used_users.append(str(us))
        self.final_queries.append(queries)
        self.num_changed_queries.append(count_change_query)
        self.percentage_used_snippets.append((1.0 - sum(queries)/ sum(queries_total)))
        self.trajectories_results.append(tot_result)
        self.gold_standards.append(gold)

        # print("Done with user: ", str(us))
        # print('final queries: ', queries)
        # print('number of changed queries:', count_change_query )
        # print('number of used snippets:', 1.0 - sum(queries)/ sum(queries_total) )
        #
        # print('our method precision', total_prec)
        # print('our results', tot_result)
        #
        # print(self.reward_matrix)

        return counter


if __name__ == "__main__":
    # ToDo Pegah: Please check the file training_script.py to run DQN
    pass
# -*- coding: utf-8 -*-
import os
import random
import sys
import numpy as np
import CODE.preprocessing as prep
import pickle
from random import shuffle, randint

from CODE.Evaluation import Evaluation
from CODE.environment import Environment
from CODE.agent import Agent
from CODE.Sars import Sars
from sklearn.externals import joblib

path_replay_memory = "/../DATA/replay_memory.pkl"
path_history = "../DATA/history"


# TODO can we have repeatables, ask Pegah ta daaaa : PA response: no it should not be repeatable samples

class DQN:

    def __init__(self, env_, agent_, list_users_):

        self.env = env_
        self.agent = agent_

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
        for i in output: #range(0, number):
            #element_list.append(ar[np.random.randint(0, len(ar))])
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

    def training_phase(self):

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
            while len(replay_memory_ar) <= 2000:

                random_user = self.list_users[randint(0, len_list_user)]
                s, pass_us = self.env.reset(random_user, is_pa=True)

                if pass_us:
                    continue

                for x in range(0, 100):  # TODO PA: why 30 times?
                    a = Sars.get_random_action_vector_pa(6)
                    r, s_prime, done = self.env.step_pa(self.agent.actions_to_take_pa(a), a)
                    replay_memory_ar.append(Sars(s, a, r, s_prime, False))
                    s = s_prime

                print('gold standars', self.env.current_name, self.env.golden_standard_db)
                print('extrcated name entities', self.env.university_name_pa, self.env.date_pa)

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

    def get_max_action(self, state):

        arg_max = []
        for i in range(6):
            action_vector = self.generate_action(i, 6)
            in_vector = np.concatenate([state, action_vector], axis=1)

            arg_max.append(self.agent.network.predict(in_vector))
        if self.bad_franky(arg_max):
            print("The project is in danger :(, out_vector ", arg_max)

        action_vector = [0] * 6
        action_vector[arg_max.index(max(arg_max))] = 1

        return action_vector

# TODO too big for it's own good, break into funcs
    def deep_QN(self):

        eps = 0.1 #TODO PA: the eps should be tested with the smaller values
        history = []

        # Desc: loading users
        #list_users = sorted(list(map(int, os.listdir(self.env.path))))

        # Desc: loading weights
        if os.path.exists(os.getcwd() + path_replay_memory):
            replay_memory = joblib.load(os.getcwd() + path_replay_memory)
        else:
            # Desc: generate first replay memory
            #MDP framework for Information Extraction (Traiing Phase)
            replay_memory = self.training_phase()

        # train episodes
        for us in self.list_users[35:36]:

            # reset episode with new user and get initial state
            # TODO PA: the gamma should be increased to 0.95 or 0.99 after the first test
            gamma = 0.9

            # initial state
            state, err = self.env.reset(us, is_pa = True)
            episode = {}
            if err:
                continue
            episode[len(history)] = []

            done = False
            # DQN with experience replace

            counter = 0
            while not done:
            #for i in range(50):

                if counter > 1000:
                    print('we use break option')
                    break

                """
                Select an action with an epsilon
                """
                p = np.random.random()

                if p < eps:
                    action_vector = Sars.get_random_action_vector(6)[0]
                else:
                    # Desc: a = argmaxQ(s,a)
                    action_vector = self.get_max_action(state)

                # Observe reward and new state
                reward, next_state, done = self.env.step_pa(self.agent.actions_to_take_pa(action_vector), action_vector)

                episode[len(history)].append(reward)

                #print("reward step:: ", reward)
                #print("current_db in agent:: ", agent.env.current_db)

                #print("Gold standard:: ", env.current_name, env.golden_standard_db)
                #print("Current queue:: ", env.current_queue)
                #print("Current snippet:: ", env.current_text)

                # Todo Ask Pegah about replay memory ask her opinion...?
                if len(replay_memory) < 1000:
                    replay_memory.append(Sars(state, action_vector, reward, next_state))
                else:
                    print("ADDING TO MEMORY...", " reward type", type(reward), " reward", reward)
                    del replay_memory[0]
                    replay_memory.append(Sars(state, action_vector, reward, next_state))

                # Desc: Q[s,a] = Q[s,a] + learning_rate*(reward + discount* max_a'(Q[s',a']) - Q[s,a])
                # this part is for learning the Q function using gradient descent
                X_train = []
                Y_train = []

                tempo = self.get_random_elements(replay_memory, 1000)

                for sample in tempo:

                    # if state is terminal
                    if self.env._check_grid() or (sample.s_prime[0, 15] == sample.s_prime[0, 16] == sample.s_prime[0, 17]):
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
            print("couter", self.get_best_entities_with_optimal_policy(us))
            print('gold standards', self.env.current_name, self.env.golden_standard_db)
            print('extracted entities', self.env.university_name_pa, self.env.date_pa)

            eval = Evaluation(self.env.golden_standard_db, self.env.university_name_pa, self.env.date_pa)
            print(eval.total_accuray())

    def get_best_entities_with_optimal_policy(self, us):

        # initial state
        state, err = self.env.reset(us, is_pa=True)

        done = False
        counter = 0

        while not done:
            # for i in range(50):

            if counter > 1000:
                print('we use break option')
                return counter

            action_vector = self.get_max_action(state)
            # Observe reward and new state
            reward, next_state, done = self.env.step_pa(self.agent.actions_to_take_pa(action_vector), action_vector)
            state = next_state

            counter += 1

        return counter


if __name__ == "__main__":

    env = Environment()

    if not os.path.exists(env.path_count_vect) or not os.path.exists(env.path_tfidf_vect):
        print("Training BOW vectors")
        prep.list_to_pickle_vectorizer(os.getcwd() + "/../DATA/")
        print("---FIT COMPLETED----")

    # TODO get the second number automatically ... env.tf_vectorizer.shape[0]
    # Desc: agent = Agent(env, (27 + 27386,))
    #len_vect = env.tf_vectorizer.transform([""]).toarray()
    #print(len(len_vect))

    agent = Agent(env, (28, )) # + len_vect.shape[1],)) # the comma is very important
    list_users = sorted(list(map(int, os.listdir(env.path))))

    dqn = DQN(env, agent, list_users)

    try:
        dqn.deep_QN()
    except KeyboardInterrupt:
        print("\n\n-----Interruption-----\nSaving weights")
        agent.network.save_weights(env.path_weights)
        print("Weights saved")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

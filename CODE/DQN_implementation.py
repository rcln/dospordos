# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import preprocessing as prep
from random import shuffle, randint
from environment import Environment
from agent import Agent
from Sars import Sars
from sklearn.externals import joblib

path_replay_memory = "/../DATA/replay_memory.pkl"


# TODO can we have repeatables, ask Pegah

def get_random_elements(ar: list, number):
    element_list = []
    for i in range(0, number):
        element_list.append(ar[np.random.randint(0, len(ar))])
    return element_list


def interpret_action(action_vector):
    num_l = np.nonzero(action_vector)
    num = num_l[0][0]
    actions_db = ("delete current_db", "add current_db", "keep current_db")
    actions_grid = ("next_snippet", "change_queue")

    print("Actions:: ", actions_grid[int(num / 3)], ";;", actions_db[num % 3])


def main(env, agent):
    eps = 0.5

    # Desc: loading users
    list_users = sorted(list(map(int, os.listdir(env.path))))

    # Desc: loading weights
    if os.path.exists(env.path_weights):
        agent.network.load_weights(env.path_weights)

    # Desc: agent.print_model()
    shuffle(list_users)
    len_list_user = len(list_users)

    if os.path.exists(os.getcwd() + path_replay_memory):
        replay_memory = joblib.load(os.getcwd() + path_replay_memory)
    else:  # Desc: generate first replay memory
        replay_memory_ar = []
        while len(replay_memory_ar) <= 999:
            random_user = list_users[randint(0, len_list_user)]
            s = env.reset(random_user)
            for x in range(0, 30):
                a = Sars.get_random_action_vector(6)
                r, s_prime, done = env.step(agent.actions_to_take(a))
                replay_memory_ar.append(Sars(s, a, r, s_prime, False))
                s = s_prime
        print("Saving replay memory")
        joblib.dump(replay_memory_ar, os.getcwd() + path_replay_memory)
        print("Saved replay memory")
        replay_memory = replay_memory_ar
    # episodes
    for us in list_users:
        # reset episode with new user and get initial state
        gamma = 0.1

        # initial state
        state = env.reset(us)
        done = False
        # DQN with experience replace
        while not done:
            """
            Select an action with an epsilon 
            """

            p = np.random.random()

            print("\nProbability for exploring: ", p, " vs epsilon: ", eps)

            if p < eps:
                action_vector = Sars.get_random_action_vector(6)
            else:
                # Desc: a = argmaxQ(s,a)
                arg_max = []
                for i in range(6):
                    action_vector = [0] * 6
                    action_vector[i] = 1
                    action_vector = np.array([action_vector])
                    in_vector = np.concatenate([state, action_vector], axis=1)

                    # print(in_vector.shape)
                    arg_max.append(agent.network.predict(in_vector))

                action_vector = [0] * 6
                action_vector[arg_max.index(max(arg_max))] = 1
                action_vector = np.array([action_vector])

                print("Q(s,a'), arg_max:: ", end="")
                interpret_action(action_vector)
                print("Q(s,a'), probability:: ", max(arg_max))
            # Observe reward and new state
            reward, next_state, done = env.step(agent.actions_to_take(action_vector))

            print("reward step:: ", reward)
            print("current_db in agent:: ", agent.env.current_db)
            print("ACTION TAKEN:: ", end="")
            interpret_action(action_vector)
            print("State:: ", env.get_state())
            print("State::  ( 7 dim vector for actual queue~ query result, "
                  "actual snippet normalized, 4 dim vector engine search, "
                  "common univ, common years, common, total gold_stand, total current, "
                  "total both, confidence ORG, confidence GPE, valid name variation ")

            print("Gold standard:: ", env.golden_standard_db)
            print("Current queue:: ", env.current_queue)
            print("Current snippet:: ", env.current_text)

            # Todo Ask Pegah about replay memory ask her opinion...?
            if len(replay_memory) < 4000:
                replay_memory.append(Sars(state, action_vector, reward, next_state))
            else:
                del replay_memory[0]
                replay_memory.append(Sars(state, action_vector, reward, next_state))

            # Desc: Q[s,a] = Q[s,a] + learning_rate*(reward + discount* max_a'(Q[s',a']) - Q[s,a])
            X_train = []
            Y_train = []
            for sample in get_random_elements(replay_memory, 1000):
                # s_prime.A = s_prime.B = s_prime.common in length or no more data(queues)
                # sample.s_prime = sample.s_prime.T
                # sample.s = sample.s.T

                if env._check_grid() or (sample.s_prime[0, 15] == sample.s_prime[0, 16] == sample.s_prime[0, 17]):
                    t = sample.r
                else:
                    target_ar = []
                    for i in range(6):
                        action_vector = [0] * 6
                        action_vector[i] = 1
                        action_vector = np.array([action_vector])
                        t_vector = np.concatenate((sample.s_prime, action_vector), axis=1)
                        target_ar.append(agent.network.predict(t_vector))
                    t = sample.r + gamma * max(target_ar)

                x_train = np.concatenate((sample.s, sample.a), axis=1)

                # TODO ERROR HERE
                """
                TypeError: 'NoneType' object is not callable
                """
                if len(X_train) == 0:
                    X_train = x_train
                else:
                    X_train = np.concatenate((X_train, x_train))

                Y_train.append(t[0])

            Y_train = np.array(Y_train)

            agent.network.fit(X_train, Y_train, 1, len(x_train))

            state = next_state

        agent.network.save_weights(env.path_weights)
        break


if __name__ == "__main__":
    env = Environment()

    if not os.path.exists(env.path_count_vect) or not os.path.exists(env.path_tfidf_vect):
        print("Training BOW vectors")
        prep.list_to_pickle_vectorizer(os.getcwd() + "/../DATA/")
        print("---FIT COMPLETED----")

    # TODO get the second number automatically ... env.tf_vectorizer.shape[0]
    # Desc: agent = Agent(env, (27 + 27386,))
    len_vect = env.tf_vectorizer.transform([""]).toarray()
    agent = Agent(env, (27 + len_vect.shape[1],)) # the comma is very important
    try:
        main(env, agent)
    except KeyboardInterrupt:
        print("\n\n-----Interruption-----\nSaving weights")
        agent.network.save_weights(env.path_weights)
        print("Weights saved")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

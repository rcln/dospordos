# -*- coding: utf-8 -*-
import os
import sys
from environment import Environment
from agent import Agent
import numpy as np


# TODO can we have repeatables
def get_random_elements(ar: list, number):
    element_list = []
    for i in range(0, number):
        element_list.append(ar[np.random.randint(0, len(ar))])
    return element_list


def get_random_action_vector(size):
    action_vector = np.zeros(size)
    action_vector[np.random.randint(0, size)] = 1
    return action_vector


def get_random_sars():
    A = np.random.uniform(0.0, 10.0)
    B = np.random.uniform(0.0, 10.0)
    a = get_random_action_vector(6)
    r = np.array([np.random.uniform(-20.0, 0.0)])
    s = np.concatenate((get_random_action_vector(7),
                        np.array([np.random.uniform(0.0, 1.0)]),
                        get_random_action_vector(4),
                        np.array([np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)]),
                        np.array((A, B, A+B)),
                        np.array([np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)]),
                        np.array([np.random.randint(0, 2)])))
    A = np.random.uniform(0.0, 10.0)
    B = np.random.uniform(0.0, 10.0)
    s_prime = np.concatenate((get_random_action_vector(7),
                        np.array([np.random.uniform(0.0, 1.0)]),
                        get_random_action_vector(4),
                        np.array([np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)]),
                        np.array((A, B, A+B)),
                        np.array([np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)]),
                        np.array([np.random.randint(0, 2)])))
    return s, a, r, s_prime


def interpret_action(action_vector):
    num_l = np.nonzero(action_vector)
    num = num_l[0][0]
    actions_db = ("delete current_db", "add current_db", "keep current_db")
    actions_grid = ("next_snippet", "change_queue")

    print("Actions:: ", actions_grid[int(num / 3)], ";;", actions_db[num % 3])


def main(env, agent):

    eps = 0.5

    # loading users
    list_users = sorted(list(map(int, os.listdir(env.path))))

    # loading weights
    if os.path.exists(env.path_weights):
        agent.network.load_weights(env.path_weights)

    # agent.print_model()

    # Todo random users

    # episodes
    for us in list_users:
        # reset episode with new user and get initial state
        replay_memory = []
        gamma = 0.1
        for x in range(0,1000):
            replay_memory.append(get_random_sars())

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
                action_vector = get_random_action_vector(6)
            else:
                # a = argmaxQ(s,a)
                arg_max = []
                for i in range(6):
                    action_vector = [0]*6
                    action_vector[i] = 1
                    in_vector = [state + action_vector]
                    in_vector = np.array(in_vector)
                    # print(in_vector.shape)
                    arg_max.append(agent.network.predict(in_vector))

                action_vector = [0]*6
                action_vector[arg_max.index(max(arg_max))] = 1

                print("Q(s,a'), arg_max:: ", end="")
                interpret_action(action_vector)
                print("Q(s,a'), probability:: ", max(arg_max))
            # Observe reward and new state
            # example
            reward, next_state, done = env.step(agent.actions_to_take(action_vector))

            print("reward:: ", reward)
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

            # Todo Ask Pegah about replay memory
            if len(replay_memory) < 500:
                replay_memory.append((state, action_vector, reward, next_state))
            else:
                del replay_memory[0]
                replay_memory.append((state, action_vector, reward, next_state))

            # Q[s,a] = Q[s,a] + learning_rate*(reward + discount* max_a'(Q[s',a']) - Q[s,a])
            X_train = []
            Y_train = []
            for sample in get_random_elements(replay_memory, 30):
                # s_prime.A = s_prime.B = s_prime.common in length or no more data(queues)
                if env._check_grid() or (sample[3][-6] == sample[3][-5] == sample[3][-4]):
                    t = sample[2]
                else:
                    target_ar = []
                    for i in range(6):
                        action_vector = [0] * 6
                        action_vector[i] = 1
                        t_vector = np.concatenate((sample[3], np.array(action_vector)))
                        t_vector = np.array([t_vector])
                        target_ar.append(agent.network.predict(t_vector))
                    t = sample[2] + gamma*max(target_ar)
                x_train = np.concatenate((sample[0], sample[1]))
                x_train = np.array(x_train)
                X_train.append(x_train)
                Y_train.append(t[0])

            X_train = np.array(X_train)
            Y_train = np.array(Y_train)

            agent.network.fit(X_train, Y_train, 1, len(x_train))

            state = next_state

        agent.network.save_weights(env.path_weights)
        break

if __name__ == "__main__":
    try:
        env = Environment()
        agent = Agent(env)
        main(env, agent)
    except KeyboardInterrupt:
        print("\n\n-----Interruption-----\nSaving weights")
        agent.network.save_weights(env.path_weights)
        print("Weights saved")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

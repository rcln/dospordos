# -*- coding: utf-8 -*-
import os
from environment import Environment
from agent import Agent
import numpy as np

# TODO can we have repeatables
def get_random_elements(ar: list,number):
    element_list = []
    for i in range(0,number):
        element_list.append(ar[np.random.randint(0,len(ar))])
    return element_list


def get_random_action_vector(size):
    action_vector = np.zeros(size)
    action_vector[np.random.randint(0,size)] = 1
    return action_vector


def get_random_sars():
    A = np.random.uniform(0.0, 10.0)
    B = np.random.uniform(0.0, 10.0)
    a = get_random_action_vector(6)
    r = np.array([np.random.uniform(-20.0, 0.0)])
    s = np.concatenate((get_random_action_vector(7),
                        np.array([np.random.uniform(0.0, 1.0)]),
                        get_random_action_vector(4),
                        np.array([np.random.uniform(0.0, 1.0),np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)]),
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


def main():

    eps = 0.1

    data_path = "".join(s+"/" for s in (os.getcwd().split('/')[:-1])) + "DATA/"
    data_train = data_path+"train_db/"
    data_db = data_path+"fer_db/train.json"

    # loading users
    list_users = sorted(list(map(int, os.listdir(data_train))))

    env = Environment()
    agent = Agent(env)

    env.set_path_train(data_train)
    env.set_path_files(data_db)

    # agent.print_model()

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
                    print(in_vector.shape)
                    arg_max.append(agent.network.predict(in_vector))

                action_vector = [0]*6
                action_vector[arg_max.index(max(arg_max))] = 1

            # Observe reward and new state
            # example
            reward, next_state, done = env.step(agent.actions_to_take(action_vector))
            print(reward)
            print(next_state)
            print(done)

            if len(replay_memory) < 40:
                replay_memory.append((state, action_vector, reward, next_state))
            else:
                del replay_memory[0]
                replay_memory.append((state, action_vector, reward, next_state))

            # Q[s,a] = Q[s,a] + learning_rate*(reward + discount* max_a'(Q[s',a']) - Q[s,a])
            # TODO cal the target with all actions
            for sample in get_random_elements(replay_memory, 10):
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
                x_train = np.array([x_train])
                agent.network.fit(x_train, np.array([t]), 1, 1)

            state = next_state
            done = True
        break

if __name__ == "__main__":
    main()

import os
from environment import Environment
from agent import Agent
import numpy as np


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
                        np.array([np.random.uniform(0.0, 1.0),np.random.uniform(0.0, 1.0),np.random.uniform(0.0, 1.0)]),
                        np.array((A, B, A+B)),
                        np.array([np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)]),
                        np.array([np.random.randint(0, 2)])))
    A = np.random.uniform(0.0, 10.0)
    B = np.random.uniform(0.0, 10.0)
    s_prime = np.concatenate((get_random_action_vector(7),
                        np.array([np.random.uniform(0.0, 1.0)]),
                        get_random_action_vector(4),
                        np.array([np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0),np.random.uniform(0.0, 1.0)]),
                        np.array((A, B, A+B)),
                        np.array([np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)]),
                        np.array([np.random.randint(0, 2)])))
    return s, a, r, s_prime


def main():

    eps = 0.1

    data_path = "".join(s+"/" for s in (os.getcwd().split('/')[:-1])) + "DATA/"
    data_train = data_path+"train_db/"
    data_db = data_path+"db_fer/train.json"

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
        # TODO fucntion to make this random
        replay_memory = []

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
                    arg_max.append(agent.network.predict(in_vector))

                action_vector = [0]*6
                action_vector[arg_max.index(max(arg_max))] = 1

                pass

            # Observe reward and new state
            # example
            # reward, next_state, done = env.step(agent.next_snippet, agent.)

            # Q[s,a] = Q[s,a] + learning_rate*(reward + discount* max_a'(Q[s',a']) - Q[s,a])
            # state = next_state
            done = True
        break

if __name__ == "__main__":
    main()

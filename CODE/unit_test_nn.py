import os
from CODE.environment import Environment
from CODE.agent import Agent
import numpy as np

"""
ValueError: Error when checking : expected input_1 to have shape (None, 27) but got array with shape (27, 1)
"""

def main():

    data_path = "".join(s+"/" for s in (os.getcwd().split('/')[:-1])) + "DATA/"
    data_train = data_path+"train_db/"
    data_db = data_path+"fer_db/train.json"

    # loading users
    list_users = sorted(list(map(int, os.listdir(data_train))))

    env = Environment()
    #agent = Agent(env)

    env.set_path_train(data_train)
    env.set_path_files(data_db)

    state = env.reset(1)
    print(state)
    # action_vector = [1,0,0,0,0,0]
    # input_vector = [state+action_vector]
    # input_vector = np.array(input_vector)
    # print(input_vector)
    # print(input_vector)
    # print(input_vector.shape)
    # # print(agent.network.summary())
    # print(agent.network.predict(input_vector))

def test():

    data_train = "../DATA/train_db/"

    # loading users
    #list_users = sorted(list(map(int, os.listdir(data_train))))

    env = Environment()

    agent = Agent(env)
    state = env.reset(1)

    print('state',state)
    print('reward', env._get_reward(), env._get_soft_reward())

    action_vector = [1,0,0,0,0,0]

    print('state', state)
    print("state", state[0][0][0:21])
    print(type(state))
    print('action', action_vector)
    print(type(action_vector))

    input_vector = [(state[0][0][0:21]).tolist() + action_vector]
    input_vector = np.asarray(input_vector)
    print(input_vector)
    print(input_vector.shape)

    print(agent.network.predict(input_vector))

    pass

# if __name__ == "__main__":
#
#     #main()
#     test()


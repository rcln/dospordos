import os
from environment import Environment
from agent import Agent
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

if __name__ == "__main__":
    main()

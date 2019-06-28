# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging

from DQN_implementation import DQN_NN
from environment import Environment
from agent import Agent
import pandas as pd
import numpy as np

# ToDo Note to Pegah
# From the terminal run this script. Selecting the folder with the data, the algorithm and is_RE
# Example:
# > python3 training_script.py ~/project/dospordos/DATA/db_v1_ns/train_db/ DQN 0
# It will run the algorithm DQN with is_RE=0 and the data is in that path
# If the directory is for testing
# > python3 training_script.py ~/project/dospordos/DATA/db_v1_ns/test_db/ -is_test=1

df = pd.read_excel('/home/pegah/Documents/research/IIMAS/dospordos/DATA/fer_db/test_traj.xlsx')
info_test_traj = np.asarray(df.values)
if __name__ == "__main__":

    parser = argparse.ArgumentParser("training_script")
    parser.add_argument("DB", help="Path to training directory")
    parser.add_argument("ALG", help="Algorithm to execute", default="DQN")
    parser.add_argument("is_RE", help="Use of Regular Expression", default=0)
    parser.add_argument("-is_test", help="The data is for testing", required=False,
                        default=0)
    parser.add_argument("-iteration_test", help="number of iterations for test", required=False,default=1)
    parser.add_argument("-given_weight", help="given weight to the NN", required=False,
                        default='')


    parser.add_argument("-is_baseline", help="Is it the baseline for the test",
                        required=False,
                        default=0)

    parser.add_argument("-is_db_v2", help="Is the second database",
                        required=False,
                        default= False)

    parser.add_argument("-initial_range", help="Initial range of users", required=False, default="-1")
    parser.add_argument("-final_range", help="Final range of users", required=False, default="-1")
    parser.add_argument("-v", "--verbose",
                        action="store_true", dest="verbose",
                        help="Verbose mode [Off]")
    args = parser.parse_args()

    if args.verbose:
        def verbose(*args):
            for a in args:
                print(a)
            print('\033[0m')

    if not os.path.isdir(args.DB):
        raise ValueError("Path doesn't exists")

    path_data = args.DB
    algorithm = args.ALG
    is_RE = args.is_RE
    is_test = args.is_test

    iteration_test = args.iteration_test
    #if is_test:
    given_weight = args.given_weight

    is_db_v2 = args.is_db_v2
    is_baseline = args.is_baseline
    initial_range = args.initial_range
    final_range = args.final_range


    name = str(algorithm) + "_" + str(is_RE) + "_" + str(path_data.split('/')[-3])
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler = logging.FileHandler("../DATA/"+name+".log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

    logger.debug('NEW RUN')

    if given_weight != '' and is_test:
        env = Environment(path=path_data, path_weights= '../DATA/'+ given_weight, is_db_v2=is_db_v2)
        name = given_weight[:-3]
    else:
        env = Environment(path=path_data, path_weights= name+'_weights.h5', is_db_v2=is_db_v2) #path_weights='../DATA/weights_0.2956216549873352_350.h5', is_db_v2=is_db_v2) #
    # ToDo Note to Pegah: for using the second data base
    if is_db_v2:
        agent = Agent(env, (25,)) # 27
    else:
        agent = Agent(env, (24,)) # 26
    # list_users = sorted(list(map(int, os.listdir(env.path))))
    list_users = os.listdir(env.path)


    if initial_range != "-1" and final_range != "-1":
        list_users = list_users[int(initial_range):int(final_range)]
    elif initial_range != "-1":
        list_users = list_users[int(initial_range):]
    elif final_range != "-1":
        list_users = list_users[:int(final_range)]

    dqn = DQN_NN(env, agent, list_users, is_RE=bool(is_RE), logger=logger, name=name)

    try:
        if is_test == "1":
            if is_baseline:
                dqn.testing_baselines(iteration_test = int(iteration_test), traj_matrix= info_test_traj)
            else:
                dqn.testing(eps=0.1, iteration_test = int(iteration_test),traj_matrix= info_test_traj)

        elif algorithm.upper() == "DQN":
            dqn.deep_QN(gamma=0.95, eps=0.1, training_replay_size= 50)#0000) #2000)
        elif algorithm.upper() == "DDQN":
            dqn.DoubleDQN(gamma=0.95, eps=0.1, training_replay_size=2000, is_db_v2= is_db_v2)

    except KeyboardInterrupt:
        print("\n\n-----Interruption-----\nSaving weights")
        print("PATH:", env.path_weights)
        agent.network.save_weights(env.path_weights)
        print("Weights saved")
        sys.exit(0)

    # just for selecting 100 users from the source
    # dqn.users_for_trajectory(size=100)
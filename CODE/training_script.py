# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging

from DQN_implementation import DQN
from environment import Environment
from agent import Agent

# ToDo Note to Pegah
# From the terminal run this script. Selecting the folder with the data, the algorithm and is_RE
# Example:
# > python3 training_script.py ~/project/dospordos/DATA/db_v1_ns/train_db/ DQN 0
# It will run the algorithm DQN with is_RE=0 and the data is in that path
# If the directory is for testing
# > python3 training_script.py ~/project/dospordos/DATA/db_v1_ns/train_db/ DQN 0 -is_test=1

if __name__ == "__main__":

    parser = argparse.ArgumentParser("training_script")
    parser.add_argument("DB", help="Path to training directory")
    parser.add_argument("ALG", help="Algorithm to execute")
    parser.add_argument("is_RE", help="Use of Regular Expression")
    parser.add_argument("-is_test", help="The data is for testing", required=False,
                        default=0)
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
    name = str(algorithm) + "_" + str(is_RE) + "_" + str(path_data.split('/')[-3])
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler = logging.FileHandler("../DATA/"+name+".log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

    logger.debug('NEW RUN')

    env = Environment(path=path_data, path_weights=name+'_weights.h5')

    agent = Agent(env, (28,))
    # list_users = sorted(list(map(int, os.listdir(env.path))))
    list_users = os.listdir(env.path)

    dqn = DQN(env, agent, list_users, is_RE=int(is_RE), logger=logger, name=name, is_test=is_test)

    try:
        if algorithm.upper() == "DQN":
            dqn.deep_QN(gamma=0.95, eps=0.1, training_replay_size=2000)
        elif algorithm.upper() == "DDQN":
            dqn.DoubleDQN(gamma=0.95, eps=0.1, training_replay_size=2000)

    except KeyboardInterrupt:
        print("\n\n-----Interruption-----\nSaving weights")
        agent.network.save_weights(env.path_weights)
        print("Weights saved")
        sys.exit(0)

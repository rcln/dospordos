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
# > python3 training_script.py ~/project/dospordos/DATA/db_v1_ns/test_db/ -is_test=1

if __name__ == "__main__":

    parser = argparse.ArgumentParser("training_script")
    parser.add_argument("DB", help="Path to training directory")
    parser.add_argument("ALG", help="Algorithm to execute", default="DQN")
    parser.add_argument("is_RE", help="Use of Regular Expression", default="0")
    parser.add_argument("-is_test", help="The data is for testing", required=False,
                        default=0)
    parser.add_argument("-is_db_v2", help="Is the second database", required=False,
                        default=0)
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
    is_db_v2 = args.is_db_v2
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

    env = Environment(path=path_data, path_weights=name+'_weights.h5', is_db_v2=is_db_v2)

    # ToDo Note to Pegah: for using the second data base
    if int(is_db_v2) == 1:
        agent = Agent(env, (29,))
    else:
        agent = Agent(env, (28,))
    # list_users = sorted(list(map(int, os.listdir(env.path))))
    list_users = os.listdir(env.path)

    if initial_range != "-1" and final_range != "-1":
        list_users = list_users[int(initial_range):int(final_range)]
    elif initial_range != "-1":
        list_users = list_users[int(initial_range):]
    elif final_range != "-1":
        list_users = list_users[:int(final_range)]

    dqn = DQN(env, agent, list_users, is_RE=int(is_RE), logger=logger, name=name)

    try:
        if is_test == "1":
            dqn.testing(eps=0.1)
        elif algorithm.upper() == "DQN":
            dqn.deep_QN(gamma=0.95, eps=0.1, training_replay_size=2000)
        elif algorithm.upper() == "DDQN":
            dqn.DoubleDQN(gamma=0.95, eps=0.1, training_replay_size=2000)

    except KeyboardInterrupt:
        print("\n\n-----Interruption-----\nSaving weights")
        print("PATH:", env.path_weights)
        agent.network.save_weights(env.path_weights)
        print("Weights saved")
        sys.exit(0)

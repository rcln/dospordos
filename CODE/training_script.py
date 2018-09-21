# -*- coding: utf-8 -*-

import os
import sys

import CODE.preprocessing as prep

from DQN_implementation import DQN
from environment import Environment
from agent import Agent


if __name__ == "__main__":
    env = Environment()

    if not os.path.exists(env.path_count_vect) or not os.path.exists(env.path_tfidf_vect):
        print("Training BOW vectors")
        prep.list_to_pickle_vectorizer(os.getcwd() + "/../DATA/")
        print("---FIT COMPLETED----")

    # TODO get the second number automatically ... env.tf_vectorizer.shape[0]
    # Desc: agent = Agent(env, (27 + 27386,))
    # len_vect = env.tf_vectorizer.transform([""]).toarray()
    # print(len(len_vect))

    agent = Agent(env, (28,))  # + len_vect.shape[1],)) # the comma is very important
    list_users = sorted(list(map(int, os.listdir(env.path))))

    dqn = DQN(env, agent, list_users, is_RE=True)

    try:
        dqn.deep_QN(gamma=0.95, eps=0.1, training_replay_size=2000)
        # dqn.DoubleDQN(gamma= 0.95, eps = 0.1, training_replay_size= 2000)

    except KeyboardInterrupt:
        print("\n\n-----Interruption-----\nSaving weights")
        agent.network.save_weights(env.path_weights)
        print("Weights saved")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)



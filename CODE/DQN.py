import logging
import random

import numpy as np

path_history = "../DATA/history"
path_model = "../DATA/dqn/model_nn.h5"

class DQN:
    """we have two DQN approaches in general: 1- DQN + normal NE 2 - DQN + normal NE + regular expresions"""

    def __init__(self, env_, agent_, list_users_, is_RE, logger, name):
        """
        :param env_:
        :param agent_:
        :param list_users_:
        :param is_RE: verifies if we use regular expressions in name entity extraction or not.
        """

        self.env = env_
        self.agent = agent_
        self.is_RE = is_RE
        self.logger = logger
        self.name = name
        self.path_replay_memory = '/../DATA/' + self.name + '_replay_memory.pkl'
        # Desc: loading users
        self.list_users = list_users_

        self.reward_matrix = []
        self.accuracy_matrix = []
        self.measure_results_matrix = []

        "new added measures by Pegah"
        self.used_users = []
        self.final_queries = []
        self.num_changed_queries = []
        self.percentage_used_snippets = []
        self.trajectories_results = []
        self.gold_standards = []

        self.base_ma_list = []
        self.base_ctg_list = []


        self.action_size = 5 #7

        # self.callbacks = [agent.EarlyStopByLossVal()]
        # ToDo: Note to Pegah, another callback with it can be stopped if there's no improvement with a min_delta .

    def get_random_elements(self, ar: list, number):
        """
        choose number of non repetitive elements from ar list.
        :param ar:
        :param number:
        :return:
        """
        output = random.sample(ar, number)
        element_list = []
        for i in output:
            element_list.append(i)
        return element_list

    def interpret_action(self, action_vector):
        num_l = np.nonzero(action_vector)
        num = num_l[1][0]

        actions_db = ("delete current_db", "add current_db", "keep current_db")
        actions_grid = ("next_snippet", "change_queue")

        print("Actions:: ", actions_grid[int(num / 3)], ";;", actions_db[num % 3])
        return

    def bad_franky(self, ar):
        return np.isnan(ar).any() or np.isinf(ar).any()


    def generate_action(self, i, length):
        action_vector = [0] * length
        action_vector[i] = 1
        action_vector = np.array([action_vector])
        return action_vector
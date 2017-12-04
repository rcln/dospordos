# -*- coding: utf-8 -*-

import json
import os
import pandas as pd
from queue import Queue


class Environment:

    def __init__(self):
        self.path = "~/dospordos/DATA/train_db/"
        self.path_db = "~/dospordos/DATA/db_fer/"
        self.queries = {}
        self.current_query = None
        self.current_data = {}
        self.data_db = None

    def set_path_files(self, path):
        self.path = path

    def set_path_data_db(self, path):
        self.path_db = path

    # start new episode
    def reset(self, id_person):

        files = self.path + str(id_person)+"/"
        if not os.path.exists(files):
            raise ValueError('path given doesn\'t exits:\n'+files)

        self.queries.clear()
        self.current_data.clear()
        self.current_query = None
        self._get_data_db()
        for file in os.listdir(files):
            with open(files+file) as f:
                data_raw = f.read()
            data = json.loads(data_raw)
            q = Queue()
            num_snippet = []
            for num in data:
                num_snippet.append(int(num))
            num_snippet = sorted(num_snippet)
            for i in num_snippet:
                q.put(data[str(i)])
            if self.current_query is None:
                self.current_query = 0
            self.queries[len(self.queries)] = q

        initial_state = self.get_state()

        return initial_state

    def step(self, action_query, action_db):

        action_query()
        action_db()

        next_state = self.get_state()
        reward = self._get_reward()
        done = self._check_grid()

        return reward, next_state, done

    def get_queries(self):
        queries = []
        for k in self.queries.keys():
            queries.append(k)
        return queries

    def get_state(self):
        state = []

        # Todo get confidence scores
        # self._get_confidence(state)

        # Todo get tf-idf..?

        # common, Total  (only Univ),   common, Total (only year),  common, Total(U-A)
        data_db = self.data_db[1]
        data_cur = self.current_data.get(1, None)
        if data_cur is None:
            state.extend([0]*6)
            return state

        # Todo compare tuples efficiently

        return state

    def _get_data_db(self):
        # todo do a query or search in fer data..?
        self.data_db = {0: 'name', 1: [('univ', 'anio')]}

    def _check_grid(self):
        empty = False

        for k in self.queries.keys():
            if self.queries[k].qsize() == 0:
                empty = True
                break
        return empty

    def _get_reward(self):
        reward = self._get_jaccard_distance()

        # todo add penalties due to size of data

        return reward

    def _get_jaccard_distance(self):
        # dj(A,B) = 1 - J(A,B)
        # J(A,B) = |A^B| / |AvB|
        # Todo take consideration of semantic similarities
        # Todo duda intersección y union de tuplas
        dist = 0
        return dist

    @staticmethod
    def _get_confidence(state): # Todo Theo's working on this
        state.append(1)






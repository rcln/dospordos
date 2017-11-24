# -*- coding: utf-8 -*-

import json
import os
from queue import Queue


class Environment:

    def __init__(self):
        self.path = "~/dospordos/DATA/train_db/"
        self.queries = {}
        self.current_query = None

    def set_path_files(self, path):
        self.path = path

    # start new episode
    def reset(self, id_person):

        initial_state = []
        files = self.path + str(id_person)+"/"
        if not os.path.exists(files):
            raise ValueError('path given doesn\'t exits:\n'+files)

        self.queries.clear()
        self.current_query = None
        for file in os.listdir(files):
            with open(files+file) as f:
                data_raw = f.read()
            data = json.loads(data_raw)
            q = Queue()
            num_snippet = []
            for num in data:
                num_snippet.append(int(num))
            num_snippet = sorted(num_snippet)
            search = ""
            for i in num_snippet:
                search = data[str(i)]['search']
                q.put(data[str(i)])
            if self.current_query is None:
                self.current_query = search
            self.queries[search] = q
        return initial_state

    def step(self, action_query, action_db):
        reward = 0
        next_state = []

        action_query()
        action_db()
        done = self.queries[self.current_query].qsize() == 0

        return reward, next_state, done

    def get_queries(self):
        queries = []
        for k in self.queries.keys():
            queries.append(k)
        return queries









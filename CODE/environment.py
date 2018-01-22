# -*- coding: utf-8 -*-

import json
import os
from queue import Queue

import spacy
from collections import defaultdict

class Environment:

    def __init__(self):
        self.path = "/dospordos/DATA/train_db/"
        self.path_db = "/dospordos/DATA/db_fer/"
        self.queries = {}
        self.current_query = None
        self.current_data = {}
        self.data_db = None

    def set_path_files(self, path):
        self.path = path

    def set_path_train(self, path):
        self.path_db = path

    # start new episode
    def reset(self, id_person):

        files = self.path + str(id_person)+"/"
        if not os.path.exists(files):
            raise ValueError('path given doesn\'t exits:\n'+files)

        self.queries.clear()
        self.current_data.clear()
        self.current_query = None
        self._get_data_db(id_person)
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

        self._get_confidence(state)

        # common, Total  (only Univ),   common, Total (only year),  common, Total(U-A)
        data_db = self.data_db[1]
        data_cur = self.current_data.get(1, None)
        if data_cur is None:
            state.extend([0]*6)
            return state

        A = set(data_db)
        B = set(data_cur)
        set_uni_A = set()
        set_ani_A = set()
        set_uni_B = set()
        set_ani_B = set()

        for y1 in data_db:
            set_uni_A.add(y1[0])
            set_ani_A.add(y1[1])

        for y2 in data_cur:
            set_uni_B.add(y2[0])
            set_ani_B.add(y2[1])

        total = len(A.union(B))
        common = len(A.intersection(B))
        commonU = len(set_uni_A.intersection(set_uni_B))
        commonA = len(set_ani_A.intersection(set_ani_B))

        state.append(commonU)
        state.append(commonA)
        state.append(common)
        state.append(total)

        return state

    def _get_data_db(self, id_person):
        if not os.path.exists(self.path_db):
            raise ValueError('path given doesn\'t exits:\n' + self.path_db)

        with open(self.path_db) as f:
            data_raw = f.read()
            tmp = json.loads(data_raw)
            tmp = tmp['_default']
        grid = [tmp[str(id_person)]['institution'], tmp[str(id_person)]['year_finish']]

        self.data_db = grid

    def _check_grid(self):
        empty = False

        for k in self.queries.keys():
            if self.queries[k].qsize() == 0:
                empty = True
                break
        return empty

    # Todo familias de equivalencia semántica
    def _get_reward(self):
        data_db = self.data_db[1]
        data_cur = self.current_data.get(1, [])
        a = set(data_db)
        b = set(data_cur)

        # Jaccard index - symmetric difference (penalty)
        reward = (len(a.intersection(b))/len(a.union(b))) - len(a.symmetric_difference(b))

        return reward

    def _get_reward_soft(self):
        data_db = self.data_db[1]
        data_cur = self.current_data.get(1, [])

        a = set()
        b = set()
        for y1 in data_db:
            a.add(y1[0])

        for y2 in data_cur:
            b.add(y2[0])

        # Jaccard index - symmetric difference (penalty)
        reward = (len(a.intersection(b)) / len(a.union(b))) - len(a.symmetric_difference(b))

        return reward

    @staticmethod
    def _get_confidence(text):
        nlp = spacy.load('en')
        ner_org = ('', u'ORG', -1.0)
        ner_gpe = ('', u'GPE', -1.0)
        with nlp.disable_pipes('ner'):
            doc = nlp(text)

        (beams, something_else_not_used) = nlp.entity.beam_parse([doc], beam_width=16, beam_density=0.0001)

        entity_scores = defaultdict(float)
        for beam in beams:
            for score, ents in nlp.entity.moves.get_beam_parses(beam):
                for start, end, label in ents:
                    entity_scores[(start, end, label)] += score

        for key in entity_scores:
            start, end, label = key
            if label == 'ORG' and entity_scores[key] > ner_org[2]:
                ner_org = (doc[start:end], label, entity_scores[key])
            elif label == 'GPE' and entity_scores[key] > ner_gpe[2]:
                ner_gpe = (doc[start:end], label, entity_scores[key])
        return ner_org, ner_gpe


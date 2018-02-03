# -*- coding: utf-8 -*-

import json
import os,re
import spacy
from queue import Queue
from collections import defaultdict

class Environment:

    def __init__(self):
        self.path = "/dospordos/DATA/train_db/"
        self.path_db = "/dospordos/DATA/fer_db/"
        self.queues = {}
        self.current_queue = None
        self.current_text = ""
        self.current_data = None
        self.current_db = []
        self.golden_standard_db = None
        self.info_snippet = None

    def set_path_train(self, path):
        self.path = path

    def set_path_files(self, path):
        self.path_db = path

    # start new episode
    def reset(self, id_person):

        files = self.path + str(id_person)+"/"
        if not os.path.exists(files):
            raise ValueError('path given doesn\'t exits:\n'+files)

        self.queues.clear()
        self.current_db.clear()
        self.current_queue = None
        self._get_golden_standard_db(id_person)
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
            if self.current_queue is None:
                self.current_queue = 0
            self.queues[len(self.queues)] = q

        self.current_data = self.queues[self.current_queue].get()

        initial_state = self.get_state()

        return initial_state

    def step(self, action_query, action_current_db, *args):

        action_query(*args)
        action_current_db()

        next_state = self.get_state()
        reward = self._get_reward()
        done = self._check_grid()

        return reward, next_state, done

    def get_queries(self):
        queries = []
        for k in self.queues.keys():
            queries.append(k)
        return queries

    def get_state(self):
        state = []


        self.current_text = self.current_data['title']+" "+self.current_data['text']

        text = self.current_text
        self._fill_info_snippet()

        # common, Total  (only Univ),   common, Total (only year),  common, Total(U-A)
        golden_standard_db = self.golden_standard_db
        data_cur = self.info_snippet

        A = set(golden_standard_db)
        B = set(data_cur)
        set_uni_A = set()
        set_ani_A = set()
        set_uni_B = set()
        set_ani_B = set()

        for y1 in golden_standard_db:
            set_uni_A.add(y1[0])
            set_ani_A.add(y1[1])

        for y2 in data_cur:
            set_uni_B.add(y2[0])
            set_ani_B.add(y2[1])

        total = len(A.union(B))
        common = len(A.intersection(B))
        commonU = len(set_uni_A.intersection(set_uni_B))
        commonA = len(set_ani_A.intersection(set_ani_B))
        state = state + self._int_to_onehot(7,self.current_queue)         #state.append(self.current_queue)
        state.append(self._normalize_snippet_number(float(self.current_data['number_snippet'])))
        state = state + self._int_to_onehot(4, self.current_queue)       #state.append(int(self.current_data['engine_search']))
        state.append(commonU)
        state.append(commonA)
        state.append(common)
        state.append(total)


        tmp_vec = self._get_confidence(text)
        for v in tmp_vec:
            state.append(v[2])

        return state



    def _get_golden_standard_db(self, id_person):
        if not os.path.exists(self.path_db):
            raise ValueError('path given doesn\'t exits:\n' + self.path_db)

        tags = ['institution', 'year_finish']
        with open(self.path_db) as f:
            data_raw = f.read()
            tmp = json.loads(data_raw)
            grid = tmp['_default']

        tmp = []
        for tag in tags:
            tmp.append(grid[str(id_person)][tag])

        self.golden_standard_db = [tuple(tmp)]

    def _check_grid(self):
        empty = False

        for k in self.queues.keys():
            if self.queues[k].qsize() == 0:
                empty = True
                break
        return empty

    # Todo familias de equivalencia semántica
    def _get_reward(self):
        golden_standard_db = self.golden_standard_db
        data_cur = self.current_db
        a = set(golden_standard_db)
        b = set(data_cur)

        # Jaccard index - symmetric difference (penalty)
        reward = (len(a.intersection(b))/len(a.union(b))) - len(a.symmetric_difference(b))

        return reward



    def _get_reward_soft(self):
        golden_standard_db = self.golden_standard_db
        data_cur = self.current_db

        a = set()
        b = set()
        for y1 in golden_standard_db:
            a.add(y1[0])

        for y2 in data_cur:
            b.add(y2[0])

        # Jaccard index - symmetric difference (penalty)
        reward = (len(a.intersection(b)) / len(a.union(b))) - len(a.symmetric_difference(b))

        return reward

    @staticmethod
    def _get_date(text, first=False):
        maches = re.findall(r'\d{4}', text)
        if first and len(maches) > 0:
            return [maches[0]]
        else:
            return maches

    def _get_location(text):
        nlp = spacy.load('en_core_web_sm')
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

        if (ner_gpe[2] >= ner_org[2]):
            return ner_gpe[0]
        else:
            return ner_org[0]

    @staticmethod
    def _get_confidence(text):
        nlp = spacy.load('en_core_web_sm')
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

    def _fill_info_snippet(self,text):
        date = self._get_date(text,True)
        location = self._get_location(text,True)
        return (location, date)

    @staticmethod
    def _int_to_onehot(length,number):
        l = [0] * length
        l.__setitem__(number-1, 1)
        return l

    def _normalize_snippet_number(self, snippet_number):
        return 1 - (snippet_number / float(len(self.queues[self.current_queue])))
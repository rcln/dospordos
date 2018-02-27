# -*- coding: utf-8 -*-

import utils
import json, os
from queue import Queue
from utils import FeatureFilter


class Environment:

    def __init__(self):
        self.path = "/dospordos/DATA/train_db/"
        self.path_db = "/dospordos/DATA/fer_db/"
        self.queues = {}
        self.current_queue = None
        self.current_text = ""
        self.current_name = ""
        self.current_data = None
        self.current_db = []
        self.golden_standard_db = None
        self.info_snippet = None

    def set_path_train(self, path):
        self.path = path

    def set_path_files(self, path):
        self.path_db = path

    # start new episode
    # there are 4518 person names in train-db
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
        # The input vector for our NN
        initial_state = self.get_state()

        return initial_state

    def step(self, action_tuple, *args):
        # action_query(*args)
        # action_current_db()

        action_tuple[0]()
        action_tuple[1]()
        next_state = self.get_state()
        reward = self._get_reward()
        done = self._check_grid() or self._is_finished()

        return reward, next_state, done

    def _is_finished(self):
        return self.current_db == self.golden_standard_db

    def get_queries(self):
        queries = []
        for k in self.queues.keys():
            queries.append(k)
        return queries

    def get_state(self):
        state = []

        self.current_text = self.current_data['title']+" "+self.current_data['text']

        text = self.current_text
        self.info_snippet = []
        #for a provided text in snippet result, we get date and organization of text if there exist any.
        self.info_snippet.append(self._fill_info_snippet(text))

        # common, Total  (only Univ),   common, Total (only year),  common, Total(U-A)
        golden_standard_db = self.golden_standard_db

        data_cur = self.info_snippet

        # print(golden_standard_db)
        # print(data_cur)

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

        #it defines which query result is taken for this state. We have 7 query types in total.
        state = state + utils.int_to_onehot(7, self.current_queue)         #state.append(self.current_queue)
        #We normalize the taken snippet number of query results w.r.t rest of query snippents results
        #PA: I do not know the reason yet.
        state.append(self._normalize_snippet_number(float(self.current_data['number_snippet'])))
        # shows the selcted engine search. There are 4 enigine searches in total.
        state = state + utils.int_to_onehot(4, int(self.current_data['engine_search']),True)       #state.append(int(self.current_data['engine_search']))
        # number of common university names between goal standards and extracted university names from the given snippet
        state.append(commonU)
        # number of common dates between goal standards and extracted dates from the given snippet
        state.append(commonA)
        # number of set of common dates and university names between goal standards and extracted dates and university names from the given snippet
        state.append(common)
        # number of total university names and dates given in goal standards
        state.append(len(A))
        # number of total university name and dates extracted from the given snippet
        state.append(len(B)) # -6
        #total number of universities names and dates in union of goal standards and the given snippet
        state.append(total)

        tmp_vec = utils.get_confidence(text)

        # get the confidence score for NER with Spacy on GPE (    ner_gpe = (GPE: Geopolitical entity, i.e. countries, cities, states)
        #TODO: PA (Question) this is a confident score for extracted entities (ORG and GPE) of given text
        for v in tmp_vec:
            state.append(v[2])

        #checks if the person name is valid or not.
        state.append(self._valid_name())

        return state

    def _valid_name(self):
        filter = FeatureFilter(self.current_name)
        if filter.has_nominal(self.current_text):
            return 1
        else:
            return 0

    def _get_golden_standard_db(self, id_person):
        if not os.path.exists(self.path_db):
            raise ValueError('path given doesn\'t exits:\n' + self.path_db)

        # PA:year_start can be taken into account too.
        tags = ['institution', 'year_finish']
        with open(self.path_db) as f:
            data_raw = f.read()
            tmp = json.loads(data_raw)
            grid = tmp['_default']

        tmp = []
        for tag in tags:
            tmp.append(grid[str(id_person)][tag])

        self.current_name = (grid[str(id_person)]['name']).strip()
        self.golden_standard_db = [tuple(tmp)]

    def _check_grid(self):
        empty = True

        for k in self.queues.keys():
            if self.queues[k].qsize() != 0:
                empty = False
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

    def _fill_info_snippet(self,text):
        date = utils.get_date(text,True)
        location = utils.get_location(text)
        return (location, date)

    def _normalize_snippet_number(self, snippet_number):
        return 1 - (snippet_number / float((self.queues[self.current_queue]).qsize()))



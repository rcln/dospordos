# -*- coding: utf-8 -*-

import json
import os
import numpy as np
from queue import Queue

import utils
from environment_core import Environment_core


class Environment:

    def __init__(self, path, path_weights, is_db_v2):
        self.env_core = Environment_core(path, path_weights, is_db_v2)
        self.path = self.env_core.path
        self.path_weights = self.env_core.path_weights
        #self.current_db = self.env_core.current_db

    # start new episode
    # there are 4518 person names in train-db
    def reset(self, id_person, is_RE, is_pa=False):
        self.env_core.person_id = id_person
        self.env_core.university_name_pa = set()
        self.env_core.date_pa = set()
        self.env_core.que_changed_obligatory = False
        self.env_core.check_university_pa = False
        self.env_core.check_date_pa = False

        files = self.env_core.path + str(id_person)+"/"

        if not os.path.exists(files):
            raise ValueError('path given doesn\'t exits:\n'+files)

        self.env_core.queues.clear()
        self.env_core.current_db.clear()
        self.env_core.current_queue = 0
        self.env_core._get_golden_standard_db(id_person)
        self.env_core.reward_prev = 0

        for t in self.env_core.golden_standard_db:
            if None in t:
                print("RESET...", self.env_core.golden_standard_db)
                return 0, True
            # self.queues is a list including several queues such that each one returned back to a set of documents
            # extracted by a query. In total self.queues for each id_person has 7 elements (equal to 7 queries)
        dir_list = os.listdir(files)
        uni_name_gs = self.env_core.golden_standard_db[0][0].lower()

        queues = {}
        POS = False

        for file in dir_list:
            with open(files + file, 'r') as f:
                data_raw = f.read()
            data = json.loads(data_raw)
            q = Queue()

            for snippet in data[file.replace('.json', '')]:
                if not (snippet['text'] or snippet['title']):
                    continue

                if len(snippet['text'].strip()) == 0 or len(snippet['title'].strip()) == 0:
                    continue
                q.put(snippet)
                if not POS:
                    if uni_name_gs in snippet['text'].lower():
                        POS = True
                    if uni_name_gs in snippet['title'].lower():
                        POS = True
            queues[len(queues)] = q
        self.env_core.queues = queues

        self.env_core.current_data = self.env_core.queues[self.env_core.current_queue].get()
        # part of the input vector for our NN
        initial_state = self.get_state(is_RE, pa_state=True)

        if POS:
            return initial_state, False
        else:
            return 0, True

    def step(self, action_tuple, *args, is_RE=0):
        #TODO PA: what is *args input here? why nothing work here?
        # Todo Answer: It was intended for a further version of the model. Where the actions
        # were capable of receiving arguments. An action would be select an specific queue
        # instead of just jumping to the next one; in order to do that you need the argument
        # besides other actions might don't receive any argument

        #TODO PA: step does not update the pointer on queires!
        # ToDo: Josue: In the previous idea, the actions were supposed to affect the environment
        # that's why the step doesn't affect the pointer directly but rather is done with an
        # specific action

        # action_query(*args)
        # action_current_db()

        action_tuple[0]()
        action_tuple[1]()

        next_state = self.get_state(is_RE)

        # Todo Find the optimal reward
        reward = self.env_core._get_reward()

        done = self.env_core._check_grid() or self.env_core._is_finished()

        return reward, next_state, done

    def step_pa(self, action_tuple, *args, is_RE=0):

        # action_query(*args)
        # action_current_db()

        previous_entities = self.env_core.info_snippet
        next_state = self.get_state(is_RE=is_RE, pa_state=True)
        reward = self.env_core._get_reward_pa(previous_entities, *args)

        done = self.env_core._check_grid() or self.env_core._is_finished_pa()

        return reward, next_state, done

    def get_state(self, is_RE, pa_state=False):

        # TODO PA: states are the vectors of size 27407, which is a high dimension vector, which the state size is 22 and the rest (27386) are related to vect_tf

        """
        this function returns back the current state which is a vector consisting of
        :return: 7 dimension vector with one 1 and the rest are zeros: indicates which query result is considered in the current state
                 the popped up snippet order from the query results w.r.t the rest of unpopped snippents in query result s.t. 0 <= the normalized value <= 1
                 another 4 dimensional e_i vector indicating which research engine is used w.r.t the four utilized research engines
                 number of common university names or organisations between goal standards and extracted NE in snippet
                 number of common dates between goal standards and extracted NE in snippet
                 number of common extracted NER in snippet and all goal standards
                 number of all goal standards
                 number of all extracted NER from the snippet
                 number of union of goal standards and NER in snippet
                 confident score for extracted ORG and GPE by Spacy from the snippet
                 if a person name given by "goal standards" is a valid person name or not
                 #TODO PA: checking the person name coming from Fernando DB as a person name sounds strange. It makes more sence checking the person name coming from the snippet as a person name.
                    # Answer:
                        It verifies whether a valid variation of the name exists in the snippet or not.
                        The class is initialized with the name in order to make the valid variations.
                        Then the method search a valid name variation in the text passed
                And a huge vector for vect_tf !!
        """
        # state = []
        text = self.current_text = self.env_core.current_data['title']+" "+self.env_core.current_data['text']

        # text = self.current_text
        self.info_snippet = []

        location_confident = utils.get_confidence(text)
        #location_confident = utils.get_confidence_RE(text)

        # for a provided text in snippet result, we get date and organization of text if there exist any.
        if pa_state:
            self.info_snippet.append(self.env_core._fill_info_snippet_pa(text, location_confident, is_RE))
        else:
            self.info_snippet.append(self.env_core._fill_info_snippet(text, location_confident[0], location_confident[1]))

        # common, Total  (only Univ),   common, Total (only year),  common, Total(U-A)
        golden_standard_db = self.env_core.golden_standard_db
        #data_cur = self.info_snippet

        tempo = self.info_snippet

        unis = list(set(tempo[0][0]))
        dates = tempo[0][1]
        data_cur = [tuple(unis+dates)]

        data_cur_dic = {'unis': unis, 'dates': dates}

        # university name and date coming from goal standards (fernando database)
        A = set(golden_standard_db)
        # B includes date and location (ORG and GPE) extracted from the given snippet
        B = set(data_cur)
        set_uni_A = set()
        set_ani_A = set()
        set_uni_B = set()
        set_ani_B = set()

        print(self.env_core.current_name, golden_standard_db, ',', data_cur)

        for y1 in golden_standard_db:
            set_uni_A.add(y1[0])
            set_ani_A.add(y1[1])
            set_ani_A.add(y1[2])

        for y2 in data_cur_dic['dates']:
            set_ani_B.add(y2)

        for y3 in data_cur_dic['unis']:
            set_uni_B.add(y3)

        total = len(A.union(B))
        common = len(A.intersection(B))
        commonU = len(set_uni_A.intersection(set_uni_B))
        commonA = len(set_ani_A.intersection(set_ani_B))

        # ToDo Note to Pegah, for the second database the one-hot of search engine has increased
        # utils.int_to_onehot(5, int(self.current_data['engine_search']), True)

        if self.env_core.is_db_v2:
            vec_engine = utils.int_to_onehot(5, int(self.env_core.current_data['engine_search']), True)
        else:
            vec_engine = utils.int_to_onehot(4, int(self.env_core.current_data['engine_search']), True)

        # it defines which query result is taken for this state. We have 7 query types in total.
        #state = text
        state = utils.int_to_onehot(7, self.env_core.current_queue) \
                + [self.env_core._normalize_snippet_number(float(self.env_core.current_data['number_snippet']))] \
                + vec_engine \
                + [commonU, commonA, common, len(A), len(B), total]
        
        # We normalize the taken snippet number of query results w.r.t rest of query snippets results
        # PA: I do not know the reason yet.
        """ Answer: The number of the current snippet have values between 0 and approximately 40, 
                since we believe the more important information in the beginning of the search, we decided
                to realize a normalization and pass a value between 0 and 1. Where 1 is the beginning of the
                snippets in the queue 
        """
        # state.append(self._normalize_snippet_number(float(self.current_data['number_snippet'])))
        # shows the selected engine search. There are 4 enigine searches in total.
        # state = state + utils.int_to_onehot(4, int(self.current_data['engine_search']), True)
        # #state.append(int(self.current_data['engine_search']))
        # number of common university names between goal standards and extracted university names from the given snippet
        # state.append(commonU)
        # number of common dates between goal standards and extracted dates from the given snippet
        # state.append(commonA)
        # number of set of common dates and university names between goal standards and extracted dates and university
        # names from the given snippet
        # state.append(common)
        # number of total university names and dates given in goal standards
        # state.append(len(A))
        # number of total university name and dates extracted from the given snippet
        # state.append(len(B)) # -6
        # total number of universities names and dates in union of goal standards and the given snippet
        # state.append(total)


        # TODO: PA (Question) this is a confident score for extracted entities (ORG and GPE) from the given text.
        # this entities get extracted again inside the get_confidence function
        # Todo Answer: Indeed the entities are extracted again, but I wanted to keep it functional and don't store those
        # values in memory; just compute them when is necessary and get the desired values
        # get the confidence score for NER with Spacy on GPE (ner_gpe = (GPE: Geopolitical entity, i.e. countries, cities, states)

        #for v in location_confident:
        #    state.append(1)

        # checks if the person name is valid or not.
        state.append(self.env_core._valid_name())

        # TODO PA: what is necessity of adding vect_tf?
        # Todo Answer: The status of the environment has information about the current status of the system standard
        # , the golden standard, the queries, and the entities. However, it doesn't have any information related with
        # the snippet text. Following the idea of LSTM, we add the tf-idf in order to have some context of the snippet
        # and then add it to the status.

        # vect_tf = self.tf_vectorizer.transform([text]).toarray()

        state = np.array([state])
        #state = np.concatenate([state, vect_tf], axis=1)

        self.env_core.info_snippet = self.info_snippet
        return state







# -*- coding: utf-8 -*-
import json
import os

# import preprocessing as prep
from queue import Queue
import utils

from sklearn.externals import joblib
import random
import re

from environment_core import Environment_core
re_clarify=re.compile("-.*")


class Environment:

    def __init__(self, path, path_weights, is_db_v2):
        self.env_core = Environment_core(path, path_weights, is_db_v2)
        self.path = self.env_core.path
        self.path_weights = self.env_core.path_weights
        #self.current_db = self.env_core.current_db

    # start new episode
    # there are 4518 person names in train-db
    def reset(self, id_person, is_RE, is_pa=False):
        # self.env_core.person_id = id_person
        # self.env_core.university_name_pa = set()
        # self.env_core.date_pa = set()
        # self.env_core.que_changed_obligatory = False
        # self.env_core.check_university_pa = False
        # self.env_core.check_date_pa = False
        #
        # files = self.env_core.path + str(id_person)+"/"
        #
        # if not os.path.exists(files):
        #     raise ValueError('path given doesn\'t exits:\n'+files)
        #
        # self.env_core.queues.clear()
        # self.env_core.current_db.clear()
        # self.env_core.current_queue = 0
        # self.env_core._get_golden_standard_db(id_person)
        # self.env_core.reward_prev = 0
        #
        # for t in self.env_core.golden_standard_db:
        #     if None in t:
        #         print("RESET...", self.env_core.golden_standard_db)
        #         return 0, True
        #     # self.queues is a list including several queues such that each one returned back to a set of documents
        #     # extracted by a query. In total self.queues for each id_person has 7 elements (equal to 7 queries)
        # dir_list=os.listdir(files)
        # uni_name_gs=self.env_core.golden_standard_db[0][0].lower()
        #
        # queues = {}
        # POS = False
        #
        # for file in dir_list:
        #     with open(files + file, 'r') as f:
        #         data_raw = f.read()
        #     data = json.loads(data_raw)
        #     q = Queue()
        #
        #     for snippet in data[file.replace('.json', '')]:
        #         if not (snippet['text'] or snippet['title']):
        #             continue
        #
        #         if len(snippet['text'].strip())==0 or len(snippet['title'].strip())==0:
        #             continue
        #         q.put(snippet)
        #         if not POS:
        #             if uni_name_gs in snippet['text'].lower():
        #                 POS = True
        #             if uni_name_gs in snippet['title'].lower():
        #                 POS = True
        #     queues[len(queues)] = q
        # self.env_core.queues = queues
        #
        # self.env_core.current_data = self.env_core.queues[self.env_core.current_queue].get()

        POS = self.env_core.pre_reset(id_person, is_RE, is_pa)

        if POS:
            # part of the input vector for our NN
            initial_state = self.get_state(is_RE, pa_state=True)
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

        next_state = self.get_state(is_RE)

        # Todo Find the optimal reward
        reward = self.env_core._get_reward()

        done = self.env_core._check_grid() or self.env_core._is_finished()

        return reward, next_state, done

    def step_pa(self, action_tuple, *args, is_RE=0):

        #action_tuple[0]()
        #action_tuple[1]()

        next_state = self.get_state(is_RE=is_RE, pa_state=True)

        #previous_entities = self.env_core.info_snippet
        #reward = self.env_core._get_reward_pa(previous_entities, *args)
        #done = self.env_core._check_grid() or self.env_core._is_finished_pa()

        (reward, done) = self.env_core.step_core(*args)

        #return reward, next_state, done
        return reward, next_state, done

    def get_state(self, is_RE, pa_state=False):
        
        #print("@@@@@@@@@@@@@@@@@@@",self.current_data,self.current_queue)
        self.env_core.current_text = self.env_core.current_data['title']+" "+self.env_core.current_data['text']
        text = self.env_core.current_text

        # text = self.current_text
        self.env_core.info_snippet = []

        location_confident = utils.get_confidence(text)
        #location_confident = utils.get_confidence_RE(text)

        # for a provided text in snippet result, we get date and organization of text if there exist any.
        if pa_state:
            self.env_core.info_snippet.append(self.env_core._fill_info_snippet_pa(text, location_confident, is_RE))
        else:
            self.env_core.info_snippet.append(self.env_core._fill_info_snippet(text, location_confident[0], location_confident[1]))

        tempo = self.env_core.info_snippet

        unis = list(set(tempo[0][0]))
        dates = tempo[0][1]

        # it defines which query result is taken for this state. We have 7 query types in total.
        state = text
        if not self.env_core.current_queue:
            q=0
        else:
            q=self.env_core.current_queue

        state_plus = [len(self.env_core.current_db),len(unis),len(dates),q,self.env_core.queues[q].qsize()]

        return state,state_plus





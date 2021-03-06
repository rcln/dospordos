# -*- coding: utf-8 -*-
import os
import operator
import random

import utils
from Evaluation import Evaluation
from environment import Environment

from agent import Agent
from regular_ne import re_organization

path_entities_memory = "/../DATA/entities_memory.pkl"


class Baselines:

    def __init__(self, env, agent, list_users, is_RE=0, is_LSTM = False):

        self.env = env
        self.is_LSTM = is_LSTM
        self.agent = agent
        self.list_users = list_users
        self.is_RE = is_RE
        self.path = "../DATA/"

    def get_entities_randomly(self, us):
        state, err = self.env.reset(us, is_RE=self.is_RE)
        queues = self.env.env_core.queues
        queue_len = len(queues)
        ques_indexes = [i for i in range(queue_len)]
        while len(ques_indexes)>0:
            index = random.choice(ques_indexes)
            ques_indexes.remove(index)
            print(index, self.env.env_core.queues[index])

            current_data = self.env.env_core.queues[index].get()
            text = current_data['title'] + " " + current_data['text']

            break

        pass

    def baseline_agregate_NE(self, user_id, is_random = False):

        snippets_vs_error = ([],[],[])

        # for user_id in self.list_users:

        # initial state
        state, err = self.env.reset(user_id, 0, is_pa=True)

        if err:
            return (0,0)

        universities_years = []
        #years = []
        names = []

        env_core = self.env.env_core
        queue_len = len(env_core.queues.keys())
        ques_indexes = [i for i in range(queue_len)]

        if is_random:
            random.shuffle(ques_indexes)

        queries_total = [self.env.env_core.queues[k].qsize() for k in self.env.env_core.queues.keys()]
        extracted_uni = set()
        extracted_years = set()

        for key in ques_indexes:
            while not env_core.queues[key].empty():
                """the queue size reduce by 1 after using the get() function"""
                current_data = env_core.queues[key].get()

                queries = [self.env.env_core.queues[k].qsize() for k in self.env.env_core.queues.keys()]

                text = current_data['title'] + " " + current_data['text']
                location_confident = utils.get_confidence(text)

                randy = random.randint(0,5)
                a = []
                if randy > 0:
                    a = a + [0]*(randy)
                a = a + [1]
                a = a + [0]*(6-randy)

                tempo = env_core._fill_info_snippet_pa(action= a, text = text,
                                                       ners= location_confident, is_RE=self.is_RE)

                _uni = [item.lower() for item in tempo[0]]
                _years = tempo[1]

                if _uni!= [] or _years!= []:
                    universities_years = universities_years + [(_uni,_years)]
                #years = years + _years

                for item in _uni:
                    extracted_uni.add(item)

                for y_ in _years:
                    extracted_years.add(y_)
                #names.append(tempo[2].lower())

                eval = Evaluation(self.env.env_core.golden_standard_db, extracted_uni, extracted_years)
                snippets_vs_error[0].append(sum(queries_total)-sum(queries))
                snippets_vs_error[1].append(eval.total_accuracy())
                snippets_vs_error[2].append(eval.get_measuring_results())

        entities = universities_years
        gold_standards = [env_core.current_name, env_core.golden_standard_db]

        # print("Saving extracted name entites in memory")
        # joblib.dump((entities, gold_standards), os.getcwd() + self.path)
        # print("Saved memory")

        #print('*****************')
        #print(entities, snippets_vs_error, gold_standards)
        return entities, snippets_vs_error, gold_standards

    def get_max_university(self, uni_dic):

        max_repeated = 0
        max_uni = ""

        for key, value in uni_dic.items():
            if value > max_repeated and key not in ['', '\n', '...', '?']:
                max_repeated = value
                max_uni = key

        return max_uni, max_repeated

    def get_max_years(self, years_dic):
        sorted_d = sorted(years_dic.items(), key=operator.itemgetter(1))
        if len(sorted_d) > 1:
            return sorted_d[-1], sorted_d[-2]
        return None

    def majority_aggregation(self, input, gold_standards):

        # if os.path.exists(os.getcwd() + path_entities_memory):
        #     (input, gold_standards) = joblib.load(os.getcwd() + path_entities_memory)
        # else:
        #     self.baseline_agregate_NE()
        #     (input, gold_standards) = joblib.load(os.getcwd() + path_entities_memory)

        person_counter = 0
        accur_uni, accu_year = (0.0, 0.0)

        # for key in input.keys():
        tempo = input
        university_repetition = {x: tempo[0].count(x) for x in tempo[0]}
        year_repetition = {x: tempo[1].count(x) for x in tempo[1]}

        max_uni, max_repeated = self.get_max_university(university_repetition)
        #print("year_repetition: ", year_repetition)
        tempo = self.get_max_years(year_repetition)

        if tempo is None:
            years = []
        else:
            years = [tempo[0][0], tempo[1][0]]

        eval = Evaluation(gold_standards[1], {max_uni}, years)
        accur = eval.total_accuracy()

        accur_uni += accur[0]
        accu_year += accur[1]

        # person_counter += 1

        return (accur_uni, accu_year)

    def closest_to_gold(self, input, gold_standards):

        golds = gold_standards[1]
        uni = golds[0][0].lower()
        years = [str(golds[0][1]), str(golds[0][2])]
        eval = Evaluation(golds, uni, years)
        #print("####>>>>",golds,uni,years)

        common_year = set()
        common_university = set()

        for y_ in input[1]:
            if y_ in years:
                common_year.add(y_)

        for u_ in input[0]:
            if u_ == uni:
                common_university.add(u_)
            elif eval.how_university(u_, uni):
                common_university.add(u_)
        #print("####>>>>",common_university,common_year)

        ev = Evaluation(golds, list(common_university), list(common_year))
        #print("####>>>>",ev.total_accuracy())

        return ev.total_accuracy()

    def filter_with_RE(self, input, gold_standards):

        entities = input[0]
        years = input[1]

        filtered_ne = set()

        for item in entities:
            if re_organization(item.title()) is not None:
                filtered_ne.add(item)

        golds = gold_standards[1]
        filtered_ne = list(filtered_ne)

        ev = Evaluation(golds, list(filtered_ne), set(list(years)))
        return ev.total_accuracy()


# if __name__ == '__main__':
#
#     env = Environment(path='../DATA/db_v1_ns/test_db/', path_weights='test.h5')
#
#     # if not os.path.exists(env.path_count_vect) or not os.path.exists(env.path_tfidf_vect):
#     #     print("Training BOW vectors")
#     #     prep.list_to_pickle_vectorizer(os.getcwd() + "/../DATA/")
#     #     print("---FIT COMPLETED----")
#
#     agent = Agent(env, (28,))  # + len_vect.shape[1],)) # the comma is very important
#     list_users = sorted(list(map(int, os.listdir(env.path))))
#
#     base = Baselines(env, agent, list_users)
#
#     for user in list_users:
#         entities, gold = base.baseline_agregate_NE(user)
#         print(base.majority_aggregation(entities, gold))
#         print(base.closest_to_gold(entities, gold))
#
#     print(base.filter_with_RE(entities, gold))

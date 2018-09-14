# -*- coding: utf-8 -*-

import json
import os
import sys
import math as m
import numpy as np
import spacy

import CODE.preprocessing as prep
from queue import Queue

from CODE import utils
from CODE.regular_ne import list_organization, re_organization
from CODE.utils import FeatureFilter
from sklearn.externals import joblib


class Environment:

    def __init__(self):
        self.path = "../DATA/train_db/"
        self.path_db = "../DATA/fer_db/train.json"
        self.path_weights = "../DATA/model_w.h5"
        self.path_count_vect = "../DATA/count_vectorizer.pkl"
        self.path_tfidf_vect = "../DATA/tfidf_vectorizer.pkl"
        self.queues = {}
        self.current_queue = None
        self.current_text = ""
        self.current_name = ""
        self.current_data = None
        self.current_db = []
        self.golden_standard_db = None
        self.info_snippet = None
        self.reward_prev = 0
        self.alpha_reward = 0.5
        self.person_id = None

        self.que_changed_obligatory = False

        self.university_name_pa = set()
        self.date_pa = set()

        self.check_university_pa = False
        self.check_date_pa = False

        # tf_vectorizer = CountVectorizer(min_df=10, stop_words='english')
        # tf_vectorizer = TfidfVectorizer(min_df=10, stop_words='english')

        if not os.path.exists(self.path_count_vect) or not os.path.exists(self.path_tfidf_vect):
            print("Training BOW vectors")
            prep.list_to_pickle_vectorizer(os.getcwd() + "/../DATA/")
            print("---FIT COMPLETED----")

        self.tf_vectorizer = joblib.load(self.path_tfidf_vect)

    def set_path_train(self, path):
        self.path = path

    def set_path_files(self, path):
        self.path_db = path

    # start new episode
    # there are 4518 person names in train-db
    def reset(self, id_person, is_RE):

        self.person_id = id_person

        self.university_name_pa = set()
        self.date_pa = set()
        self.que_changed_obligatory = False

        self.check_university_pa = False
        self.check_date_pa = False

        files = self.path + str(id_person)+"/"
        if not os.path.exists(files):
            raise ValueError('path given doesn\'t exits:\n'+files)

        self.queues.clear()
        self.current_db.clear()
        self.current_queue = 0
        self._get_golden_standard_db(id_person)
        self.reward_prev = 0

        for t in self.golden_standard_db:
            if None in t:
                print("RESET...", self.golden_standard_db)
                return 0, True

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

            # self.queues is a list including several queues such that each one returned back to a set of documents
            # extracted by a query. In total self.queues for each id_person has 7 elements (equal to 7 queries)

            self.queues[len(self.queues)] = q

        self.current_data = self.queues[self.current_queue].get()
        # part of the input vector for our NN
        initial_state = self.get_state(is_RE, pa_state= True)

        return initial_state, False

    def step(self, action_tuple, *args, is_RE):
        #TODO PA: what is *args input here? why nothing work here?
        #TODO PA: step does not update the pointer on queires!

        # action_query(*args)
        # action_current_db()

        action_tuple[0]()
        action_tuple[1]()

        next_state = self.get_state(is_RE)

        # Todo Find the optimal reward
        reward = self._get_reward()

        done = self._check_grid() or self._is_finished()

        return reward, next_state, done

    def step_pa(self, action_tuple, *args, is_RE):

        # action_query(*args)
        # action_current_db()

        previous_entities = self.info_snippet
        next_state = self.get_state(is_RE= is_RE, pa_state = True)
        reward = self._get_reward_pa(previous_entities, *args)

        done = self._check_grid() or self._is_finished_pa()

        return reward, next_state, done

    def _is_finished(self):
        return self.current_db == self.golden_standard_db

    def _is_finished_pa(self):
        #TODO PA: this is a very tough stopping criteria, we can make it easier to stop the loop.
        #print(self.check_date_pa, self.check_university_pa)

        return self.check_date_pa and self.check_university_pa

    def get_queries(self):
        queries = []
        for k in self.queues.keys():
            queries.append(k)
        return queries

    def get_state(self, is_RE, pa_state = False):

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
        state = []
        self.current_text = self.current_data['title']+" "+self.current_data['text']

        text = self.current_text
        self.info_snippet = []

        location_confident = utils.get_confidence(text)
        #location_confident = utils.get_confidence_RE(text)

        # for a provided text in snippet result, we get date and organization of text if there exist any.
        if pa_state:
            self.info_snippet.append(self._fill_info_snippet_pa(text, location_confident, is_RE))
        else:
            self.info_snippet.append(self._fill_info_snippet(text, location_confident[0], location_confident[1]))

        # common, Total  (only Univ),   common, Total (only year),  common, Total(U-A)
        golden_standard_db = self.golden_standard_db
        #data_cur = self.info_snippet

        tempo = self.info_snippet

        unis = list(set(tempo[0][0]))
        dates = tempo[0][1]
        data_cur = [tuple(unis+dates)]

        data_cur_dic = {'unis':[], 'dates': []}
        data_cur_dic['unis'] = unis
        data_cur_dic['dates'] = dates

        # university name and date coming from goal standards (fernando database)
        A = set(golden_standard_db)
        # B includes date and location (ORG and GPE) extracted from the given snippet
        B = set(data_cur)
        set_uni_A = set()
        set_ani_A = set()
        set_uni_B = set()
        set_ani_B = set()

        #print(self.current_name, golden_standard_db, ',', data_cur)

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

        # it defines which query result is taken for this state. We have 7 query types in total.
        state = state + utils.int_to_onehot(7, self.current_queue)         # state.append(self.current_queue)
        # We normalize the taken snippet number of query results w.r.t rest of query snippets results
        # PA: I do not know the reason yet.
        """ Answer: The number of the current snippet have values between 0 and approximately 40, 
                since we believe the more important information in the beginning of the search, we decided
                to realize a normalization and pass a value between 0 and 1. Where 1 is the beginning of the
                snippets in the queue 
        """
        state.append(self._normalize_snippet_number(float(self.current_data['number_snippet'])))
        # shows the selceted engine search. There are 4 enigine searches in total.
        state = state + utils.int_to_onehot(4, int(self.current_data['engine_search']), True)       #state.append(int(self.current_data['engine_search']))
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
        # total number of universities names and dates in union of goal standards and the given snippet
        state.append(total)

        # TODO: PA (Question) this is a confident score for extracted entities (ORG and GPE) from the given text.
        # this entities get extracted again inside the get_confidence function
        tmp_vec = location_confident #utils.get_confidence(text)

        # get the confidence score for NER with Spacy on GPE (ner_gpe = (GPE: Geopolitical entity, i.e. countries, cities, states)
        for v in tmp_vec:
            state.append(v[2])

        # checks if the person name is valid or not.
        state.append(self._valid_name())

        # TODO PA: what is necessity of adding vect_tf?
        vect_tf = self.tf_vectorizer.transform([text]).toarray()

        state = np.array([state])
        #state = np.concatenate([state, vect_tf], axis=1)

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

        #TODO PA:year_start can be taken into account too.
        tags = ['institution', 'year_start', 'year_finish']
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
    # TODO: PA: why are rewards defined in this way? this is completely depends on the goal standards
    # and not the improvement on one state w.r.t the previous visited state.
    def _get_reward(self, offset=3):
        golden_standard_db = self.golden_standard_db

        data_cur = []

        if golden_standard_db[0][0] is None:
            print("THE GOLD STANDARD IS MORE LIKE SILVER...[?] HMMM")
            print(self.current_data)
            print(self.golden_standard_db)
            try:
                sys.exit(-1)
            except SystemExit:
                os._exit(-2)
        else:
            tmp = golden_standard_db[0][0].lower().replace(' ', '')

        golden_standard_db = [(tmp, golden_standard_db[0][1])]

        """
        data_cur.append((tup[0][0].lower().replace(' ', ''), tup[0][1]))AttributeError: 'spacy.tokens.span.Span' object has no attribute 'lower'
        """

        for tup in self.current_db:
            data_cur.append((str(tup[0][0]).lower().replace(' ', ''), tup[0][1]))

        a = set(golden_standard_db)

        if len(a) == 0:
            print("Well josue, the world is weird")
            try:
                print("ERROR IN THE FUNCTION _get_reward()")
                sys.exit(-1)
            except SystemExit:
                os._exit(-2)

        # TODO: PA: it shouldn't be the extracted NER from the snippet in self.current_data ?
        b = set(data_cur)

        # Jaccard index - penalty
        # penalty =  e^(alpha * len(b)) * u(len(b)-offset) + min (edit_distance(A,B)) / len(A_content)
        edit_vect = np.array(utils.edit_distance(a, b))  # Range: [0, inf)

        penalty = m.pow(m.e, self.alpha_reward * len(b))*utils.step(len(b) - offset)
        penalty += edit_vect.mean() / utils.len_content(a)
        reward_cur = (len(a.intersection(b))/len(a.union(b))) - penalty

        reward = reward_cur - self.reward_prev
        self.reward_prev = reward_cur

        return reward

    # Todo rewrite this function so is similar to the normal reward but only with the universities
    def _get_reward_pa(self, previous_entities, *args):

        reward = 0

        tempo = self.que_changed_obligatory

        golden_standard_db = self.golden_standard_db
        golden_standard_db_ = ["", "", "", ""]

        action = args[0]

        if golden_standard_db[0][0] is None:
            print("THE GOLD STANDARD IS MORE LIKE SILVER...[?] HMMM")
            try:
                sys.exit(-1)
            except SystemExit:
                os._exit(-2)
        else:
            golden_standard_db_ = list(golden_standard_db[0]) + [self.current_name]

        new_entities = self.info_snippet
        # if the taken action is query
        if action[-1] == 1:
            reward = -1.0
        #if the taken action is conciling the extrcated entities
        else:
            reward = reward + self.get_accuracy(new_entities[0], previous_entities[0], golden_standard_db_)
            if self.que_changed_obligatory:
                reward += -1.0
                self.que_changed_obligatory = False

        return reward

    def _get_soft_reward(self, tolerance=3):
        """
        Only for experiential use, not used in DQN
        :param tolerance: 
        :return: 
        """

        golden_standard_db = self.golden_standard_db
        data_cur = self.current_db

        a = set()
        b = set()
        for y1 in golden_standard_db:
            a.add(y1[0])

        for y2 in data_cur:
            b.add(y2[0])

        tolerance = len(b) - tolerance
        if tolerance < 0:
            tolerance = 0

        # Jaccard index - symmetric difference (penalty)
        reward = (len(a.intersection(b)) / len(a.union(b))) - \
                 len(a.symmetric_difference(b))
        return reward

    def _fill_info_snippet(self, text, ner_org, ner_gpe):

        date = utils.get_date(text, True)
        # location = utils.get_location(text)
        if ner_gpe[2] >= ner_org[2]:
            location = ner_gpe[0]
        else:
            location = ner_org[0]

        #print("ENTITIES FOUND", location, date, " ... FOR the text... ", text)
        return str(location), str(date)

    def _fill_info_snippet_pa(self, text, ners, is_RE):

        #it returns a list of dates
        date = utils.get_date(text, False)
        location = []

        ner_org = ners[0] # organisation information (a list)
        ner_gpe = ners[1] # geographical position information (a list)
        ner_person_name = ners[2]
        #ner_date = ners[3]

        """use gazettees from Jorge work"""
        Gazettee_university = list(set(list_organization(text)))
        for item in Gazettee_university:
            location.append(item)

        location.append(str(ner_gpe[0]))
        location.append(str(ner_org[0]))

        """use RE check"""
        if is_RE:
            filtered_ne = set()
            for item in location:
                if re_organization(item.title()) is not None:
                    filtered_ne.add(item)

            location = list(filtered_ne)

        #print("ENTITIES FOUND", location, date, " ... FOR the text... ", text)
        return [list(set(location)), date, str(ner_person_name[0])] #, str(ner_date[0])

    def _normalize_snippet_number(self, snippet_number):
        try:
            div = float((self.queues[self.current_queue]).qsize())
            if div == 0:
                div = 0.001
            return 1 - (snippet_number / div)
        except KeyError:
            print("ERROR in next_snippet\n current queue: ", self.current_queue)
            print("Queues ", self.queues)
            print("DATA ", self.current_data)

    def get_accuracy(self, new_entities, previous_entities, golden_standards):

        #TODO PA: this function should be more optimised

        accuracy_prev = 0.0
        accuracy_curr = 0.0

        years = [str(golden_standards[1]) , str(golden_standards[2])]

        university_name = str(golden_standards[0].lower())

        # un_curr = str(new_entities[0].lower())
        # un_prev = str(previous_entities[0].lower())

        un_curr = [str(i).lower() for i in new_entities[0]]
        for item in un_curr:
            self.university_name_pa.add(item)
        un_prev = [str(i).lower() for i in previous_entities[0]]

        # if year is correct
        for y_ in new_entities[1]:
            self.date_pa.add(y_)
            if y_ in years:
                accuracy_curr += 0.25
                #self.date_pa.add(y_)

        if set(self.date_pa) == set(years):
            self.check_date_pa = True

        for y__ in previous_entities[1]:
            if y__ in years:
                accuracy_prev += 0.25
                #self.date_pa.add(y__)

        # if university name is correct
        if self.is_similar_university(un_curr, university_name):
            accuracy_curr += 0.5
        elif self.how_university(un_curr, university_name):
            accuracy_curr += 0.2
            #self.university_name_pa = un_curr

        #if un_prev == university_name or self.is_similar_university(un_prev, university_name):
        if self.is_similar_university(un_prev, university_name):
            accuracy_prev += 0.5
        elif self.how_university(un_prev, university_name):
            accuracy_prev += 0.2

        reward = accuracy_curr - accuracy_prev

        #add a negative reward to each step
        reward += -0.1

        return reward

    # TODO PA: Spacy depends on capital letters, we should find a way to solve this problem, thats the reason that all NEs can't be extracted.
    def how_university(self, given_list, univrsity_name):

        document_words = univrsity_name.split()

        for given in given_list:
            given_words = given.split()
            common = set(document_words).intersection(set(given_words))
            if 'university' in common:
                common.remove('university')
            if 'of' in common:
                common.remove('of')
            if 'the' in common:
                common.remove('the')
            if 'college' in common:
                common.remove('college')
            if len(common) >= 1:
                #self.university_name_pa.add(given)
                #self.check_university_pa = True
                return True

        return False

    def edit_distance(self, given_list, univrsity_name):

        return

    def is_similar_university(self, given_list, university_name):

        for item in given_list:
            if item == university_name:
                #self.university_name_pa.add(item)
                self.check_university_pa = True
                return True

        return False






import json
import re
import os
import sys
import math as m
import numpy as np

from utils import FeatureFilter
import utils
from regular_ne import list_organization, re_organization

re_clarify=re.compile("-.*")

class Environment_core:

    def __init__(self, path='../DATA/train_db/', path_weights='../DATA/model_w.h5', is_db_v2=0):
        # self.path = "../DATA/train_db/"
        self.path = path
        self.path_db = "../DATA/fer_db/train.json"
        self.fer_db = self._get_gs()
        # self.path_weights = "../DATA/model_w.h5"
        self.path_weights = "../DATA/"+path_weights
        # self.path_count_vect = "../DATA/count_vectorizer.pkl"
        # self.path_tfidf_vect = "../DATA/tfidf_vectorizer.pkl"
        self.queues = {}
        self.prev_state = ""
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
        self.is_db_v2 = is_db_v2

    def set_path_train(self, path):
        self.path = path

    def set_path_files(self, path):
        self.path_db = path

    def get_all_snippets(self):

        snippets=[]
        for id_person in os.listdir(self.path):
            files = self.path + str(id_person)+"/"

            if not os.path.exists(files):
                raise ValueError('path given doesn\'t exits:\n'+files)

            dir_list=os.listdir(files)

            for file in dir_list:
                #print("file ::", file)
                with open(files + file, 'r') as f:
                    data_raw = f.read()
                data = json.loads(data_raw)
                for snippet in data[file.replace('.json', '')]:
                    if len(snippet['title'].strip())>0 and len(snippet['text'].strip())>0:
                        snippets.append(snippet['title'].lower())
                        snippets.append(snippet['text'].lower())

        return snippets

    def _get_golden_standard_db(self, id_person):
        if not os.path.exists(self.path_db):
            raise ValueError('path given doesn\'t exits:\n' + self.path_db)

        #TODO PA:year_start can be taken into account too.
        tags = ['institution', 'year_start', 'year_finish']

        tmp = []
        for tag in tags:
            tmp.append(self.fer_db[str(id_person)][tag])

        self.current_name = (self.fer_db[str(id_person)]['name']).strip()
        self.golden_standard_db = [tuple(tmp)]

        return

    def _get_gs(self):
        with open(self.path_db, 'r') as f:
            data_raw = f.read()
            tmp = json.loads(data_raw)
            grid = tmp['_default']
            grid_= {}
            for ii,item in grid.items():
                if not item['institution']:
                    new_val=None
                else:
                    new_val=re_clarify.sub("",item['institution']).strip()
                item['institution']=new_val
                grid_[ii]=item
        return grid_

    def _is_finished(self):
        return self.current_db == self.golden_standard_db

    def _is_finished_pa(self):
        #TODO PA: this is a very tough stopping criteria, we can make it easier to stop the loop.

        # if self.check_university_pa:
        #     print("********************check university is true**********************************")
        #     print("******************************************************")
        #     print("******************************************************")
        #
        # if self.check_date_pa:
        #     print("************************check date is true******************************")
        #     print("******************************************************")
        #     print("******************************************************")

        return self.check_date_pa or self.check_university_pa

    def get_queries(self):
        queries = []
        for k in self.queues.keys():
            queries.append(k)
        return queries

    def _valid_name(self):
        filter = FeatureFilter(self.current_name)
        if filter.has_nominal(self.current_text):
            return 1
        else:
            return 0

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
            #print(self.current_data)
            #print(self.golden_standard_db)
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

    # Todo rewrite this function so is similar to the normal reward but only with the universities
    def _get_reward_pa(self, previous_entities, *args):

        reward = 0

        golden_standard_db = self.golden_standard_db
        # golden_standard_db_ = ["", "", "", ""] # Todo PA: This var is not necessary due to the if

        action = args[0]

        if golden_standard_db[0][0] is None:
            print("THE GOLD STANDARD IS MORE LIKE SILVER...[?] HMMM")
            sys.exit(-1)
        else:
            golden_standard_db_ = list(golden_standard_db[0]) + [self.current_name]

        new_entities = self.info_snippet
        # if the taken action is query
        if action[-1] == 1:
            reward = -10.0 #-1.0
        #if the taken action is conciling the extrcated entities
        else:
            reward = reward + self.get_accuracy(new_entities[0], previous_entities[0], golden_standard_db_)
            if self.que_changed_obligatory:
                #reward += -1.0
                self.que_changed_obligatory = False

        return reward

    def get_accuracy(self, new_entities, previous_entities, golden_standards):

        # TODO PA: this function should be more optimised

        accuracy_prev = 0.0
        accuracy_curr = 0.0

        years = [str(golden_standards[1]), str(golden_standards[2])]

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
                accuracy_curr += 0.1 #10.0
                #self.date_pa.add(y_)

        if set(self.date_pa) == set(years):
            self.check_date_pa = True

        for y__ in previous_entities[1]:
            if y__ in years:
                accuracy_prev += 0.1#10.0
                #self.date_pa.add(y__)

        # if university name is correct
        if self.is_similar_university(un_curr, university_name):
            accuracy_curr += 1.0 #100.0
        elif self.how_university(un_curr, university_name):
            accuracy_curr += 1.0 #100.0
            #self.university_name_pa = un_curr

        #if un_prev == university_name or self.is_similar_university(un_prev, university_name):
        if self.is_similar_university(un_prev, university_name):
            accuracy_prev += 1.0 #100.0
        elif self.how_university(un_prev, university_name):
            accuracy_prev += 1.0 #100.0

        reward = accuracy_curr - accuracy_prev

        #add a negative reward to each step
        reward -= 0.1
        #reward += -0.1*len(new_entities[0])
        #reward += -0.1*len(new_entities[1])

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

    def is_similar_university(self, given_list, university_name):
        if university_name in given_list:
            self.check_university_pa = True
            return True

        return False

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

    def _fill_info_snippet(self, text, ner_org, ner_gpe):

        date = utils.get_date(text, True)
        # location = utils.get_location(text)
        if ner_gpe[2] >= ner_org[2]:
            location = ner_gpe[0]
        else:
            location = ner_org[0]

        # print("ENTITIES FOUND", location, date, " ... FOR the text... ", text)
        return str(location), str(date)

    def _fill_info_snippet_pa(self, text, ners, is_RE):

        #it returns a list of dates
        date = utils.get_date(text, False)
        location = []

        #ner_org = ners[0] # organisation information (a list)
        #ner_gpe = ners[1] # geographical position information (a list)
        #ner_person_name = ners[2]
        #ner_date = ners[3]

        """use gazettees from Jorge work"""
        Gazettee_university = list(set(list_organization(text)))
        for item in Gazettee_university:
            location.append(item)

        #location.append(str(ner_gpe[0]))
        #location.append(str(ner_org[0]))

        """use RE check"""
        #if is_RE:
        #    filtered_ne = set()
        #    for item in location:
        #        if re_organization(item.title()) is not None:
        #            filtered_ne.add(item)

        #    location = list(filtered_ne)

        return [list(set(location)), date, None] #, str(ner_date[0])
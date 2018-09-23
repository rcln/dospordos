# -*- coding: utf-8 -*-

import editdistance

class Evaluation:

    def __init__(self, gold_standards, university_names, years):

        self.gold_standards = gold_standards
        self.university_names = university_names
        self.years = years

    def how_university(self, given, university_name):
        document_words = university_name.split()

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
        if 'imperial' in common:
            common.remove('imperial')
        if len(common) >= 1:
            return True

        return False

    def total_accuracy(self):
        percentage_uni = 0.0
        percentage_years = 0.0

        check = False

        gold_uni_name = self.gold_standards[0][0].lower()
        gold_years = [str(self.gold_standards[0][1]), str(self.gold_standards[0][2])]

        for uni in self.university_names:
            if uni == gold_uni_name:
                percentage_uni = 1.0
                check = True
                break

        if not check:
            for uni in self.university_names:
                if self.how_university(uni, gold_uni_name):
                    percentage_uni = 0.5

        for y_ in self.years:
            if y_ in gold_years:
                percentage_years += 0.5

        return percentage_uni, percentage_years

    @staticmethod
    def _prf(tp, fp, fn):
        try:
            pres = tp / (tp + fp)
        except ZeroDivisionError:
            pres = 0.0
        try:
            reca = tp / (tp + fn)
        except ZeroDivisionError:
            reca = 0.0
        try:
            fsco = 2 * pres * reca / (pres + reca)
        except ZeroDivisionError:
            fsco = 0.0
        return pres, reca, fsco

    def get_measuring_results(self):
        accuracy_uni, accuracy_years = self.total_accuracy()

        gs = []
        sys = []

        for uni in self.university_names:
            sys.append(('u', uni.lower()))
        for y_ in self.years:
            sys.append(('y', str(y_)))
        gold_uni_name = self.gold_standards[0][0].lower()
        gold_years = [str(self.gold_standards[0][1]), str(self.gold_standards[0][2])]
        gs.append(('u', gold_uni_name))
        for gy in gold_years:
            gs.append(('y', gy))

        tp, TP = 0, 0
        fp, FP = 0, 0
        fn, FN = 0, 0

        scores = []

        gs_ = list(gs)
        tp, fp, fn = 0, 0, 0

        for sv_ in sys:
            flag_tp = False
            for ii, sv in enumerate(gs_):
                print(sv, "___", sv_)
                if sv[0] == sv_[0] and editdistance.eval(sv[1], sv_[1]) / len(sv[1]) < 0.2:
                    tp += 1
                    TP += 1
                    flag_tp = True
                    gs_.append(sv)
                    del gs_[ii]
                    break
            if flag_tp:
                continue
            fp += 1
            FP += 1
        fn += len(gs_)
        FN += len(gs_)

        scores.append(self._prf(tp, fp, fn))

        P, R, F = self._prf(TP, FP, FN)

        return accuracy_uni, accuracy_years, P, R, F


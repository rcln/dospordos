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

        uni_gs = []
        uni_sys = []
        year_gs = []
        year_sys = []

        gold_uni_name = self.gold_standards[0][0].lower()
        gold_years = [str(self.gold_standards[0][1]), str(self.gold_standards[0][2])]

        for uni in self.university_names:
            uni_sys.append(('u', uni.lower()))

        uni_gs.append(('u', gold_uni_name))

        for y_ in self.years:
            year_sys.append(('y', str(y_)))

        for gy in gold_years:
            year_gs.append(('y', gy))

        TP = 0
        FP = 0
        FN = 0

        gs_ = list(uni_gs)

        for sv_ in uni_sys:
            flag_tp = False
            for ii, sv in enumerate(gs_):
                if sv[0] == sv_[0] and editdistance.eval(sv[1], sv_[1]) / len(sv[1]) < 0.2:
                    TP += 1
                    flag_tp = True
                    gs_.append(sv)
                    del gs_[ii]
                    break
            if flag_tp:
                continue
            FP += 1
        FN += len(gs_)

        Pu, Ru, Fu = self._prf(TP, FP, FN)

        TP = 0
        FP = 0
        FN = 0

        gs_ = list(year_gs)

        for sv_ in year_sys:
            flag_tp = False
            for ii, sv in enumerate(gs_):
                if sv[0] == sv_[0] and editdistance.eval(sv[1], sv_[1]) / len(sv[1]) < 0.2:
                    TP += 1
                    flag_tp = True
                    gs_.append(sv)
                    del gs_[ii]
                    break
            if flag_tp:
                continue
            FP += 1
        FN += len(gs_)

        print(self.university_names, self.years)
        Py, Ry, Fy = self._prf(TP, FP, FN)

        print('****************Pu, Ru, Fu, Py, Ry, Fy', Pu, Ru, Fu, Py, Ry, Fy)

        return Pu, Ru, Fu, Py, Ry, Fy


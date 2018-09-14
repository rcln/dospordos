#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Ivan Vladimir Meza Ruiz 2018
# GPL 3.0
#


# imports
import argparse
import os.path
import re
import editdistance

#TODO: area under the curve (AUC) should be added as another measure too.

def verbose(*args):
    return None


class fc:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ORANGE = '\033[101m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class bc:
    BLACK = '\033[40m'
    RED = '\033[41m'
    GREEN = '\033[42m'
    YELLOW = '\033[43m'
    BLUE = '\033[44m'
    MAGENTA = '\033[45m'
    CYAN = '\033[46m'
    WHITE = '\033[47m'
    RESET = '\033[49m'


class style:
    BRIGHT = '\033[1m'
    DIM = '\033[2m'
    NORMAL = '\033[22m'
    RESET = '\033[0m'


re_slot = re.compile('(?:(?P<att>[^:]+):)?(?P<val>.*)$')


def make_tuple(line):
    m = re_slot.match(line)
    if m:
        return tuple([x.lower() for x in m.groups()])
    else:
        return None

def read_slots_file(filename):
    DOCS = []
    slots = []
    nline=1
    for line in open(filename):
        line = line.strip()
        if len(line) == 0:
            DOCS.append(slots)
            slots = []
            continue
        slot_value=make_tuple(line)
        if slot_value:
            slots.append(slot_value)
        else:
            verbose(bc.YELLOW, "Not slot in line: ",nline)
        nline+=1
    if len(slots)>0:
        DOCS.append(slots)
    return DOCS

def prf(tp,fp,fn):
    try:
        pres=tp/(tp+fp)
    except ZeroDivisionError:
        pres=0.0
    try:
        reca=tp/(tp+fn)
    except ZeroDivisionError:
        reca=0.0
    try:
        fsco=2*pres*reca/(pres+reca)
    except ZeroDivisionError:
        fsco=0.0
    return pres, reca, fsco


if __name__ == '__main__':
    p = argparse.ArgumentParser("evalie")
    p.add_argument("GS",
                   help="Goldstandard slot values")
    p.add_argument("SYS",
                   help="System slot values")
    p.add_argument("-v", "--verbose",
                   action="store_true", dest="verbose",
                   help="Verbose mode [Off]")

    args = p.parse_args()
    if args.verbose:
        def verbose(*args):
            for a in args:
                print(a, end="", sep="")
            print(style.RESET)
    GS = read_slots_file(args.GS)
    SYS = read_slots_file(args.SYS)

    verbose(fc.HEADER, "GS > ", fc.ENDC, "Total docs :", len(GS))
    verbose(fc.WARNING, "SYS> ", fc.ENDC, "Total docs :", len(SYS))

    verbose(fc.HEADER, "GS > ", fc.ENDC, "Total slot-values :",
            sum([len(x) for x in GS]))
    verbose(fc.WARNING, "SYS> ", fc.ENDC, "Total slot-values :",
            sum([len(x) for x in SYS]))

    tp, TP = 0, 0
    fp, FP = 0, 0
    fn, FN = 0, 0

    scores = []

    for gs, sys in zip(GS, SYS):
        gs_ = []
        sys_ = []
        tp, fp, fn = 0, 0, 0

        for sv_ in sys:
            flag_tp = False
            for sv in gs:
                if sv[0] == sv_[0] and editdistance.eval(sv[1],sv_[1])/len(sv[1]) < 0.2:
                    tp += 1
                    TP += 1
                    flat_tp = True
                    gs_.append(sv)
                    break
            if flag_tp:
                continue
            fp += 1
            FP += 1
        fn += len(gs)-len(gs_)
        FN += len(gs)-len(gs_)

        scores.append(prf(tp, fp, fn))

    P, R, F = prf(TP, FP, FN)
    print("MACRO SCORES")
    print("Precision:", P)
    print("Recall:", R)
    print("F-score:", F)
    print("MICRO SCORES")
    print("Precision:", sum([P for P,R,F in scores])/len(scores))
    print("Recall:", sum([R for P,R,F in scores])/len(scores))
    print("F-score:", sum([F for P,R,F in scores])/len(scores))




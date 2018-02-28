# -*- coding: utf-8 -*-
import nltk, re, spacy
import numpy as np
from collections import defaultdict


class NameParser:
    """
    GRAMMAR:
        NC -> Name Complete
        N -> Name
        A -> Surname
        AA -> Common Name
        AG -> Typo Name
        I -> Initial
        AD -> Particle Name
        AAA -> Main Last Name

        Terminal nodes in the Chompsky Normal Form:
        I , AA, AG, AD

    """

    def __init__(self):
        self.grammar_head = """
                                    NC -> N A
                                    N -> N I
                                    N -> N AA
                                    N -> I
                                    N -> AA
                                    N -> AG
                                    N -> AA AD
                                    A -> AAA
                                    A -> AAA AA
                                    A -> AAA AD
                                    A -> AAA AG
                                    A -> AAA I
                                    AAA -> AA
                                    AAA -> AG
                                    AAA -> AD
                                    """
        self.name_tokenizer_regex = r'(Mc[A-Z][a-z]+|O\'[A-Z][a-z]+|[Dd]e\s[Ll]a\s[A-Z][a-z]+|-' \
                                    r'[Dd]e-[A-Z][a-z]+|[A-Z][a-z]+-[A-Z][a-z]+|[Dd]e\s[Ll]a' \
                                    r'\s[A-Z][a-z]+|[Vv][oa]n\s[A-Z][a-z]+|[Dd]e[l]?\s[A-Z][a-z]' \
                                    r'+|[A-Z][\.\s]{1,1}|[A-Z][a-z]+|[Dd]e\s[Ll]os\s[A-Z][a-z]+)'

        self.regexp_tagger_list = [(r'([A-Z][a-z]+-[A-Z][a-z]+|-[Dd]e-[A-Z][a-z]+)', 'AG'),
                                   (r'[A-Z][a-z]+', 'AA'),
                                   (r'[A-Z][\.\s]{1,1}', 'I'),
                                   (r'([A-Z][a-z]+-[A-Z][a-z]+|[Dd]e\s[Ll]a\s[A-Z][a-z]+|[Vv][oa]n'
                                    r'\s[A-Z][a-z]+|[Dd]e[l]?\s[A-Z][a-z]+|[Dd]e\s[Ll]os\s[A-Z][a-z]+)',
                                    'AD'),
                                   (r'(Mc[A-Z][a-z]+|O\'[A-Z][a-z]+)', 'AA')]

        self.tokenizer = nltk.RegexpTokenizer(self.name_tokenizer_regex)
        self.tagger = nltk.RegexpTagger(self.regexp_tagger_list)

    def parse(self, name):
        tokens = self.tokenizer.tokenize(name)
        tag_tokens = self.tagger.tag(tokens)
        terminals = ''
        for ts in tag_tokens:
            terminals += ts[1] + " -> " + "'" + ts[0] + "'" + "\n    "
        grammar_rules = self.grammar_head + terminals
        grammar = nltk.CFG.fromstring(grammar_rules)
        parser = nltk.ChartParser(grammar)
        return parser.parse(tokens)


class FeatureFilter:

    def __init__(self, base_name):
        self.base_name = base_name

    def get_nominal_vector(self, snippet):
        nominal = NominalFilter(self.base_name)
        return nominal.filter(snippet)

    def has_nominal(self, snippet):
        nominal = NominalFilter(self.base_name)
        vector = nominal.filter(snippet)
        return np.count_nonzero(vector) > 0


class NominalFilter:
    def __init__(self, name):
        name = Cleaner.remove_accent(Cleaner(), name)
        name = Cleaner.clean_reserved_xml(name)
        self.tree = NameParser.parse(NameParser(), name)
        self.list_regex = self._name_variations()
        self.dic_vect = {'L': 0, 'C': 1, 'E': 2, 'X': 3, 'V': 4}

    @staticmethod
    def _add_variation(reg, variation, label):

        if variation.get(label) is None:
            variation[label] = []

        if reg not in variation.get(label):
            variation.get(label).append(reg)

    @staticmethod
    def _compression(cn, la, initials):

        list_reg = []

        name = ""
        lastname = ""

        partition = '[\s-]+?'
        partition2 = '[\s]+?'
        for a in la:
            if lastname == "":
                lastname = a
            else:
                lastname += partition + a

        partition = '\.?[\s]*?,?'

        for n in cn:
            n = n[0]
            if name == "":
                name = n
            else:
                name += partition + n
            list_reg.append(n + partition + la[0])
            list_reg.append(la[0] + partition2 + n + partition)
            list_reg.append(n + partition + lastname)
            list_reg.append(lastname + partition2 + n + partition)

        list_reg.append(name + partition + la[0])
        list_reg.append(la[0] + partition2 + name + partition)
        list_reg.append(name + partition + lastname)
        list_reg.append(lastname + partition2 + name + partition)

        for i in initials.get('N'):
            name += partition + i

        for i in initials.get('A'):
            name += partition + i

        list_reg.append(name + partition + la[0])
        list_reg.append(la[0] + partition2 + name + partition)
        list_reg.append(name + partition + lastname)
        list_reg.append(lastname + partition2 + name + partition)

        for reg in list_reg:
            yield reg

    @staticmethod
    def _expansion(en, la, initials):

        list_reg = []

        name = ""
        lastname = ""

        partition = "[a-z]+"
        partition2 = "[\s-]+?"

        for n in en:
            if name == "":
                name = n
            else:
                name += ' ' + n
        for n in initials.get('N'):
            n = n.replace('.', '')
            name += ' ' + n + partition

        for a in la:
            if lastname == "":
                lastname = a
            else:
                lastname += partition2 + a

        for a in initials.get('A'):
            a = a.replace('.', '')
            lastname += ' ' + a + partition

        list_reg.append(name + ' ' + lastname)

        for reg in list_reg:
            yield reg

    @staticmethod
    def _inversion(vcn, la, initials):
        list_reg = []

        lastname = ""
        name = ""
        cname = ""
        partition = ',?[\s]*?'
        partition2 = '[\s-]+?[a-z]*?[\s-]+?'
        partition3 = '\.?-?'
        partition4 = '\.?'

        for a in la:
            if lastname == "":
                lastname = a
            else:
                lastname += partition2 + a

            for n in vcn:
                if name == "":
                    name = n[0]
                else:
                    name += partition3 + n[0]

                list_reg.append(lastname + partition + name + partition4)

            name = ""

        for reg in list_reg:
            yield reg

    @staticmethod
    def _literal(ln, la, initials):

        list_reg = []

        partition = '[\s-]+?'
        partition2 = ',?[\s]*?'
        partition_xtra = '(\s[a-z]+\s)+?'
        name = ""
        lastname = ""

        for a in la:
            if lastname == "":
                lastname = a
            else:
                lastname += partition + a

            for n in ln:
                if name == "":
                    name = n
                else:
                    name += partition2 + n

                list_reg.append(partition_xtra + lastname + partition_xtra)
                list_reg.append(name + partition2 + lastname + partition_xtra)
                list_reg.append(lastname + partition2 + name)

            for n in initials.get("N"):
                name += partition + n
                list_reg.append(name + partition2 + lastname)
                list_reg.append(lastname + partition2 + name)
                for ia in initials.get("A"):
                    lastname += partition + ia
                    list_reg.append(name + partition2 + lastname)
                    list_reg.append(lastname + partition2 + name)
            name = ""

        for reg in list_reg:
            yield reg

    @staticmethod
    def _extra_element(ln, la, initials):

        list_reg = []

        name = ""
        lastname = ""

        partition = '[A-Z][a-z]*.?'
        partition2 = '[\s-]+?'
        partition3 = '[\s\.,]+?'

        for a in la:
            if lastname == "":
                lastname = a
            else:
                lastname += partition2 + a

        for n in ln:
            if name == "":
                name = n
            else:
                name += partition2 + n

            list_reg.append(n + partition2 + partition + partition2 + lastname)

        list_reg.append(name + partition2 + partition + partition2 + lastname)

        for i in initials.get('N'):
            name += partition2 + i.replace('.', '') + partition3

        for i in initials.get('A'):
            name += partition2 + i.replace('.', '') + partition3

        list_reg.append(name + partition2 + partition + partition2 + lastname)

        for reg in list_reg:
            yield reg

    def _name_variations(self):
        variations = {}
        names = []
        surnames = []
        initials = {"N": [], "A": []}

        for tree in self.tree:
            for person in tree:
                if person.label() == 'N':
                    for node in person.subtrees(lambda k: k.height() == 2):
                        if node.label() == "AA":
                            names.append(
                                node.leaves()[0])
                        elif node.label() == 'I':
                            (initials.get('N')).append(
                                node.leaves()[0])
                        elif node.label() == 'AD':
                            tmp = node.leaves()[0]
                            names.append(
                                tmp.split(' ')[1]
                            )
                        elif node.label() == 'AG':
                            tmp = node.leaves()[0]
                            for t in tmp.split('-'):
                                names.append(t)

                elif person.label() == 'A':
                    for node in person.subtrees(lambda k: k.height() == 2):
                        if node.label() == 'AA':
                            surnames.append(
                                node.leaves()[0])
                        elif node.label() == 'I':
                            (initials.get('A')).append(
                                node.leaves()[0])
                        elif node.label() == 'AD':
                            tmp = node.leaves()[0]
                            surnames.append(
                                tmp.split(' ')[1]
                            )
                        elif node.label() == 'AG':
                            tmp = node.leaves()[0]
                            for t in tmp.split('-'):
                                surnames.append(t)
            break

        if surnames.__len__() > 0:
            for reg in self._literal(names, surnames, initials):
                self._add_variation(reg, variations, 'L')

            for reg in self._compression(names, surnames, initials):
                self._add_variation(reg, variations, 'C')

            for reg in self._expansion(names, surnames, initials):
                self._add_variation(reg, variations, 'E')

            for reg in self._extra_element(names, surnames, initials):
                self._add_variation(reg, variations, 'X')

            for reg in self._inversion(names, surnames, initials):
                self._add_variation(reg, variations, 'V')

            surnames = list(map(lambda x: x.upper(), surnames))
            for reg in self._literal(names, surnames, initials):
                self._add_variation(reg, variations, 'L')

            for reg in self._compression(names, surnames, initials):
                self._add_variation(reg, variations, 'C')

            for reg in self._expansion(names, surnames, initials):
                self._add_variation(reg, variations, 'E')

            for reg in self._extra_element(names, surnames, initials):
                self._add_variation(reg, variations, 'X')

            for reg in self._inversion(names, surnames, initials):
                self._add_variation(reg, variations, 'V')

            names = list(map(lambda x: x.upper(), names))
            for reg in self._literal(names, surnames, initials):
                self._add_variation(reg, variations, 'L')

            for reg in self._compression(names, surnames, initials):
                self._add_variation(reg, variations, 'C')

            for reg in self._expansion(names, surnames, initials):
                self._add_variation(reg, variations, 'E')

            for reg in self._extra_element(names, surnames, initials):
                self._add_variation(reg, variations, 'X')

            for reg in self._inversion(names, surnames, initials):
                self._add_variation(reg, variations, 'V')

        return variations

    def filter(self, snippet):

        # replace all alpahbets with accent from the snippet with without accent alphabet versions
        snippet = Cleaner.remove_accent(Cleaner(), snippet)
        snippet = Cleaner.clean_reserved_xml(snippet)

        vector = np.zeros(self.list_regex.__len__())

        for item in self.list_regex.items():
            label = item[0]
            for pattern in item[1]:
                regex = re.compile(pattern)

                if regex.search(snippet):
                    vector[self.dic_vect.get(label)] = 1

        return vector


class Cleaner:
    def __init__(self):
        self._re_a = re.compile(u'[áâàä]')
        self._re_e = re.compile(u'[éèêëě]')
        self._re_i = re.compile(u'[íïîì]')
        self._re_o = re.compile(u'[óòôöø]')
        self._re_u = re.compile(u'[úùüû]')
        self._re_n = re.compile(u'[ñ]')
        self._re_c = re.compile(u'[ç]')
        self._re_y = re.compile(u'[ỳýÿŷ]')
        self._re_beta = re.compile(u'[ß]')
        self._re_A = re.compile(u'[ÁÀÄÂÅ]')
        self._re_E = re.compile(u'[ÉÈÊË]')
        self._re_I = re.compile(u'[ÍÌÏÎ]')
        self._re_O = re.compile(u'[ÓÒÔÖØ]')
        self._re_U = re.compile(u'[ÚÙÛÜ]')
        self._re_N = re.compile(u'[Ñ]')
        self._re_C = re.compile(u'[Ç]')
        self._re_S = re.compile(u'[Š]')

    # replace all alpahbets with accent from the snippet with without accent alphabet versions
    def remove_accent(self, line_u):
        line_u = self._re_a.subn('a', line_u)[0]
        line_u = self._re_e.subn('e', line_u)[0]
        line_u = self._re_i.subn('i', line_u)[0]
        line_u = self._re_o.subn('o', line_u)[0]
        line_u = self._re_u.subn('u', line_u)[0]
        line_u = self._re_n.subn('n', line_u)[0]
        line_u = self._re_c.subn('c', line_u)[0]
        line_u = self._re_y.subn('y', line_u)[0]
        line_u = self._re_beta.subn('ss', line_u)[0]
        line_u = self._re_A.subn('A', line_u)[0]
        line_u = self._re_E.subn('E', line_u)[0]
        line_u = self._re_I.subn('I', line_u)[0]
        line_u = self._re_O.subn('O', line_u)[0]
        line_u = self._re_U.subn('U', line_u)[0]
        line_u = self._re_N.subn('N', line_u)[0]
        line_u = self._re_C.subn('C', line_u)[0]
        line_u = self._re_S.subn('S', line_u)[0]

        return line_u

    @staticmethod
    def clean_reserved_xml(line):
        r = line.replace('&apos;', "'")
        r = r.replace('&lt;', "<")
        r = r.replace('&gt;', ">")
        r = r.replace('&quot;', '"')
        r = r.replace('&amp;', "&")
        return r


def int_to_onehot(length,number,zero_based=False):
    l = [0] * length
    if zero_based:
        number-=1
    l.__setitem__(number, 1)
    return l

def get_confidence(text):
    """
    :param text:gets snippet as an input
    :return: gives back the organization name or geopolitical name such as countery, cities etc. The Spacy library extracts
     these name entities with various confidence scores. The function output is (confident score for ORG, confident score for GPE)
    using SPacy library have different
    """
    nlp = spacy.load('en_core_web_sm')
    ner_org = ('', u'ORG', 0.0) # ORG: Companies, agencies, institutions.
    ner_gpe = ('', u'GPE', 0.0) # GPE: Geopolitical entity, i.e. countries, cities, states.
    with nlp.disable_pipes('ner'):
        "convert text to spacy.tokens.doc.Doc type"
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


#there is no need to use get_location.
def get_location(text):
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


def get_date(text, first=False):
    matches = re.findall(r'\d{4}', text)
    if first and len(matches) > 0:
        return matches[0]
    elif len(matches) == 0:
        return ""
    else:
        return matches
# -*- coding: UTF-8 -*-
import re

reg_org=[
re.compile("[A-Z][a-z'\-]{3,} Univer[sz]i[td][yaéeäà][dt]?e?"),
re.compile("[A-Z\&][a-z'\-]{3,} College"),
re.compile("University.*"),
re.compile("Univer[sz]i[td][yaéeäà][dt]?e? (at|of|de|del|do|for|in|für|da|der|des|degli|della|d')"),
re.compile("[A-Z][a-z'\-]{3,} In?stit[uü]t[a-z]*"),
re.compile("In?stit[uü]t[a-z]* (of|for|für|de|on|in|d'|der|di|per)"),
re.compile("Univer[sz]i[td][yaéeäà][dt]?e? [A-Z\&]+[a-z'\-]{3,}"),
re.compile("[A-Z]+[a-z'\-]+ [ZC]ent[er][eour]m?"),
re.compile("[A-Za-z'\-]+(al|an|ische|ic) Univer[sz]i[td][yaéeäà][dt]?e?"),
re.compile("(State|College|Gakuin|City|Prefectural|Catholic) Univer[sz]i[td][yaéeäà][dt]?e?"),
re.compile("[ZC]ent[er][eour]m? (of|for|für|de|d')"),
re.compile("In?stit[uü]t[a-z]* [A-Z][a-z]{3,}"),
re.compile("Research (Institute|Council|Foundation|Center|Network|Organization|Laborator[yi]|Agency|Association|Hospital|Station|Board)"),
re.compile("Univer[sz]i[td][yaéeäà][dt]?e? (Nacional|College|Aut[óo]noma|Federal|Estadual|System|Central|Privada|Internacional|Mayor|Pontificia|Center)"),
re.compile("(Junior|University|State|City) College"),
re.compile("[A-Z\&]+[a-z'\-]{3,} Group"),
re.compile("[A-Z][a-z'\-]{3,} (H[oô]s?pital|Ospedale|[Kk]rankenhaus)"),
re.compile("[A-Z][a-z']{3,} A[ck]ad[eéè]m[yi][aec]?n?"),
re.compile("[A-Za-z'\-]+(al|an|ische|ic) College"),
re.compile("[ZC]ent[er][eour]m? [A-Z]+[a-z'\-]+"),
re.compile("Univer[sz]i[td][yaéeäà][dt]?e? [A-Za-z]+(ana|ica|ina)(\b|[\.,;])"),
re.compile("[A-Z\&]+[a-z]* As{1,2}ocia[tzcç][iaãe][oós]n?e?"),
re.compile("A[ck]ad[eéè]m[yi][aec]?n? (of|for|für|de|d'|di|der|des)"),
re.compile("[A-Z][a-z\-]{3,} Cou?n[cs][ie][ljig][olhl]?[io]?o?"),
re.compile("Organi[szt]{1,2}a[tczçs][a-z]+ (de|dos|van|do|degli|della|dels|des|of|pro|pour|per|for|für|para|d')"),
re.compile("[A-Z]+[a-z'\-]* Organi[szt]{1,2}a[tczçs][a-z]+"),
re.compile("[A-Z]+[a-z'\-]+ F[ou]{1,2}nda[tczç][iaã]on?e?"),
re.compile("Organi[szt]{1,2}a[tczçs][a-z]+ [A-Z]+[a-z'\-]*"),
re.compile("[A-Z][a-z\-]{3,} Agen[cz][yie]a?"),
re.compile("Cou?n[cs][ie][ljig][olhl]?[io]?o? (of|for|de|d'|di|della)"),
re.compile("[A-Z]+ Laborat[oó][ri][iyr][eou]?[ms]?"),
re.compile("Laborat[oó][ri][iyr][eou]?[ms]? (of|for|für|de|d'|di|national|pour|des)"),
re.compile("[A-Z]+[a-z'\-]+ Laborat[oó][ri][iyr][eou]?[ms]?"),
re.compile("[A-Z]+[a-z'\-]+ Laborat[oó][ri][iyr][eou]?[ms]?"),
re.compile("[A-za-z']+(al|an|ische|ic) Academy"),
re.compile("Agen[cz][yie]a? [A-Z][a-z\-]{3,}"),
re.compile("[A-Z][a-z\-]{3,} Society"),
re.compile("(University|Memorial|General|Centre) H[oô]spital"),
re.compile("[ZC]ent[er][eour]m? [A-Z]+[a-z]+ (of|for|für|de)"),
re.compile("[A-Z]+ In?stit[uü]t[a-z]*"),
re.compile("(H[oô]s?pital|Ospedale|[Kk]rankenhaus) [A-Z][a-z'\-]{3,}"),
re.compile("F[ou]{1,2}nda[tczç][iaã]on?e? [A-Z]+[a-z'\-]+"),
re.compile("A[ck]ad[eéè]m[yi][aec]?n? [A-Z][a-z]{3,}"),
re.compile("Laborat[oó][ri][iyr][eou]?[ms]? [A-Z]+[a-z'\-]+"),
re.compile("[A-Z][a-z'\-]+[fF]orschung"),
re.compile("[A-Z][a-z'\-]{3,} Observato[ri][iyr][oue]?m?"),
re.compile("Gro?up[oe] [A-Z\&]+[a-z'\-]{3,}"),
re.compile("(State|Medical|Military) Academy"),
re.compile("As{1,2}ocia[tzcç][iaãe][oós]n?e? (de|dos|do|degli|della|dels|des|of|pro|pour|per|for|für|para|d')"),
re.compile("College [A-Z\&]+[a-z'\-]+"),
re.compile("[A-Z\&][a-z'\-]+ Hochschule"),
re.compile("Cou?n[cs][ie][ljig][olhl]?[io]?o? [A-Z][a-z\-]+"),
re.compile("[ÉE]cole (nationale|sup[ée]rieure|normale|centrale|polytechnique|des Mines|d'ing[ée]nieurs)"),
re.compile("Hochschule [A-Z\&][a-z'\-]+"),
re.compile("F[ou]{1,2}nda[tczç][iaã]on?e? (of|for|für|de|para|pour)"),
re.compile("[A-Z][a-z\-]+institut"),
re.compile("Observato[ri][iyr][oue]?m? [A-Z][a-z'\-]{3,}"),
re.compile("Hospital(ier)? Universita(ire|rio)"),
re.compile("Fachhochschule"),
re.compile("Bundes(amt|anstalt)"),
re.compile("[A-Z][A-Z\&]+ Univer[sz]i[td][yaéeäà][dt]?e?"),
re.compile("[A-Z\&]+[a-z]* As{1,2}ocia[tzcç][iaãe][oós]n?e? (of|for|de|para|pour|d'|di|della|per)"),
re.compile("As{1,2}ocia[tzcç][iaãe][oós]n?e? [A-Za-z'\-]+ (of|for|de|para|pour|d')"),
re.compile("Agen[cz][yie]a? (for|para|pour)"),
re.compile("[A-Z][a-z\-]+ [CK]onservato[ri][riy][ueo]?m?"),
re.compile("Cou?n[cs][ie][ljig][olhl]?[io]?o? Na[ctz]ionale?"),
re.compile("Facul[td][ya]d?e? (de|of|d')"),
re.compile("[CK]onservato[ri][riy][ueo]?m? (for|de|na[tc]ional)"),
re.compile("[A-Z][a-z\-]+[ \-][Ss]tiftung"),
re.compile("[ÉE]cole d.*(ture|ique)"),
re.compile("Univer[sz]i[td][yaéeäà][dt]?e? \"[A-Za-z ]+\""),
re.compile("Colegio (de|de la) [A-Z][a-z]+"),
re.compile("[CK]onservato[ri][riy][ueo]?m? [A-Z][a-z\-]+"),
re.compile("Groupe d[e'] ?(Recherche|[EÉ]tude)"),
re.compile("[A-Z][a-z'\-]* [A-Z][a-z'\-]*organisation"),
re.compile("[A-Z\&]+[a-z'\-]+ Labs"),
re.compile("Laborat[oó][ri][iyr][eou]?[ms]? [A-Z\&]{3,}"),
re.compile("[A-Z][a-z'\-]+sternwarte"),
re.compile("[A-Z][a-z'\-]{3,} Facul[td][ya]d?e?"),
re.compile("Colegio Nacional"),
re.compile("Hochschule für"),
re.compile("[Hh]aute [ÉE]cole"),
re.compile("[A-Z][a-z'\-]*organisation [A-Z][a-z'\-]"),
re.compile("[A-Z][a-z\-]+[Uu]niversität"),
re.compile("[A-Z][a-z'\-]* [A-Z][a-z'\-]*organisation"),
re.compile("U[MP]R [0-9]+"),
re.compile("FRE [0-9]+"),
]
import time

with open("../DATA/uninames_final.txt",'r') as fd:
    words_organizations=[]
    for line in fd.readlines():
        words_organizations.append(" "+line.strip().lower()+" ")
        words_organizations.append(" "+line.strip().lower())
        words_organizations.append(line.strip().lower()+" ")


with open("../DATA/gs_instiutions.txt",'r') as fd:
    words_organizations=[]
    for line in fd.readlines():
        words_organizations.append(" "+line.strip().lower()+" ")
        words_organizations.append(" "+line.strip().lower())
        words_organizations.append(line.strip().lower()+" ")



def re_organization(text):
    for r_o in reg_org:
        m = r_o.search(text)
        if m:
            return m.group(0)
    return None

def list_organization(text):
    result=[]
    text=text.lower()

    for w in words_organizations:
        if w in text:
            result.append(w.strip())
            #print(w.strip())
            #print(text)

    return result

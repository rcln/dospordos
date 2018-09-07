class Evaluation:

    def __init__(self,gold_standards, university_names, years):

        self.gold_standards = gold_standards
        self.university_names = university_names
        self.years = years

    def how_university(self, given, univrsity_name):
        document_words = univrsity_name.split()

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


    def total_accuray(self):
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

        return (percentage_uni, percentage_years)
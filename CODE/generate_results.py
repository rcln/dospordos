import operator
import os
import pickle
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

name = "DQN_0_db_v1_ns_new"
num_iteration = 1

measure_results_matrix = []
reward_matrix = []
base_ctg_list = []
base_ma_list = []
accuracy_matrix = []
final_queries = []
num_changed_queries = []
percentage_used_snippets = []
trajectories = []
snippets_vs_error = []

"AVERAGE ON REWARDS EVOLUTIONS FOR USERS OF THE TEST SET"
def average_reward(_matrix):
    max_len = max([len(item) for item in _matrix])
    average_ = []
    for iter in range(max_len):

        counter = 0
        summ = 0

        for item in _matrix:
            if len(item) <= iter:
                summ += item[-1]
            else:
                summ += item[iter]

            counter += 1

        average_.append(summ/counter)

    return average_

def average_accuracy(_matrix):

    max_len = max([len(item) for item in _matrix])
    average_uni = []
    average_year= []
    for iter in range(max_len):

        counter = (0, 0,0,0,0,0)
        summ = (0,0,0,0,0,0)

        for item in _matrix:
            counter = tuple(map(operator.add, counter, (1, 1,1,1,1,1)))
            #print("****", item)
            if len(item) <= iter:

                summ = tuple(map(operator.add, summ, item[-1]))

            else:
                summ = tuple(map(operator.add, summ, item[iter]))

        average_uni.append(summ[2]/counter[2])
        average_year.append(summ[5]/counter[5])

    return (average_uni, average_year)


"because we have 5 iterations"
for k in range(num_iteration):

    snippets_vs_error.append(pickle.load( open( "../DATA/" + name + "_snippets_vs_error_" + str(k) + ".pkl", "rb" ) ))
    measure_results_matrix.append(pickle.load( open( "../DATA/" + name + "_mrm_" + str(k) + ".pkl", "rb" ) ))
    reward_matrix.append(pickle.load( open( "../DATA/" + name + "_rm_" + str(k) + ".pkl", "rb")))
    base_ctg_list.append(pickle.load( open( "../DATA/" + name + "_ctg_" + str(k) + ".pkl", "rb" ) ))
    base_ma_list.append(pickle.load( open("../DATA/" + name + "_ma_" + str(k) + ".pkl", "rb" ) ))
    accuracy_matrix.append(pickle.load( open( "../DATA/" + name + "_acc_" + str(k) + ".pkl", "rb" ) ))

    final_queries.append(pickle.load(open('../DATA/' + name + '_queries_' + str(k) + '.pkl', 'rb')))

    num_changed_queries.append(pickle.load(open('../DATA/' + name + '_nc_queries_' + str(k) + '.pkl', 'rb')))

    percentage_used_snippets.append(pickle.load(open('../DATA/' + name + '_per_snippets_' + str(k) + '.pkl', 'rb')))


    trajectories.append(pickle.load(open('../DATA/' + name + '_trajectories_' + str(k) + '.pkl', 'rb')))


used_users = pickle.load(open('../DATA/' + name + '_uu_' + str(0) + '.pkl', 'rb'))
gold_standards = pickle.load( open('../DATA/' + name + '_gold_standards_' + str(0) + '.pkl', 'rb'))


print("used users in test :: ", len(used_users))
print("**********************")
print("average on total exact accuracy")
print("university exact accuracy:: ", mean( [ mean([item[-1][0] for item in accuracy_matrix[i]]) for i in range(num_iteration) ] ) )
print("year exact accuracy:: ", mean( [ mean([item[-1][1] for item in accuracy_matrix[i]]) for i in range(num_iteration) ] ) )

print("**********************")
print("average precision, recall, F1 score")
print("Pu, Ru, Fu, Py, Ry, Fy")
print("university precision:: ", mean( [ mean([item[0][0] for item in measure_results_matrix[i]]) for i in range(num_iteration) ] ) )
print("university recall:: ", mean( [ mean([item[0][1] for item in measure_results_matrix[i]]) for i in range(num_iteration) ] ) )
print("university F1 score:: ", mean( [ mean([item[0][2] for item in measure_results_matrix[i]]) for i in range(num_iteration) ] ) )

print("year precision:: ", mean( [ mean([item[0][3] for item in measure_results_matrix[i]]) for i in range(num_iteration) ] ) )
print("year recall:: ", mean( [ mean([item[0][4] for item in measure_results_matrix[i]]) for i in range(num_iteration) ] ) )
print("year F1 score:: ", mean( [ mean([item[0][5] for item in measure_results_matrix[i]]) for i in range(num_iteration) ] ) )


def make_equiv(t, b, error):
    """
    for getting the exact accuracy from the t and error vectors equivalent to the b vector
    :param t:
    :param b:
    :param error:
    :return:
    """
    error_new = []
    counter= 0
    error_new.append(error[counter])
    while t[counter]<=len(b) and counter < len(t)-1:
        if t[counter+1] != t[counter]:
            error_new.append(error[counter+1])
        counter += 1

    return error_new

def get_subset_(base, err):

    base_len = len(base)-1
    new_vector_base = []
    new_vector_err = []

    if len(base)!= len(err):
        #print("******", len(base), len(err))
        #print(err)
        return False

    for i in range(0, 101, 10):

        index = round(i/100 * base_len)
        new_vector_err.append(err[index])
        new_vector_base.append(base[index])

    return new_vector_base, new_vector_err

def generate_equiv_erros(snippets_vs_errors_, base_ctg_, is_random = False):

    count = 0

    eqq_erros = []
    users = len(base_ctg_[0])


    if is_random:
        eqq_erros = [base_ctg_[0][i][1] for i in range(users)]
    else:
        for i in range(users):
            eqq_erros.append(make_equiv(snippets_vs_errors_[0][i][0], base_ctg_[0][i][0], snippets_vs_errors_[0][i][1]))

    new_vector_errors_ = []
    new_vector_base = []

    for i in range(users):
        tempo = get_subset_(base_ctg_[0][i][0], eqq_erros[i])
        if tempo:
            new_vector_base.append(tempo[0])
            new_vector_errors_.append(tempo[1])
        else:
            count += 1

    average_base = average_reward(new_vector_base)
    average_unis, average_years = average_accuracy(new_vector_errors_)

    print('//////', count)
    return average_base, average_unis, average_years


# snippets_vs_error = [snippets_vs_error[0][0:5]]
# base_ctg_list = [base_ctg_list[0][0:5]]

#av1, av2, av3 = generate_equiv_erros(snippets_vs_error, base_ctg_list)
#av1_ran, av2_rand, av3_rand = generate_equiv_erros(snippets_vs_error, base_ctg_list, is_random= True)

ll = len(base_ctg_list[0])
tt = average_accuracy([snippets_vs_error[0][i][2] for i in range(ll)])

which = 8

plt.plot(average_reward([snippets_vs_error[0][i][0] for i in range(ll)]), tt[0], label = 'uni_error')
plt.plot(average_reward([snippets_vs_error[0][i][0] for i in range(ll)]), tt[1], label = 'year_error')
#print([item[0] for item in snippets_vs_error[0][which][2] ])
#print([item[3] for item in snippets_vs_error[0][which][2] ])

#plt.plot(snippets_vs_error[0][which][0],[item[0] for item in snippets_vs_error[0][which][2] ], label = 'uni_error')
#plt.plot(snippets_vs_error[0][which][0], [item[3] for item in snippets_vs_error[0][which][2] ], label = 'year_error')

#plt.legend()
#plt.show()

toto = average_accuracy([base_ctg_list[0][i][2] for i in range(ll)])
plt.plot(average_reward([base_ctg_list[0][i][0] for i in range(ll)]), toto[0], label = 'uni_error_rand')
plt.plot(average_reward([base_ctg_list[0][i][0] for i in range(ll)]), toto[1], label = 'year_error_rand')

#plt.plot(base_ctg_list[0][which][0], [item[0] for item in base_ctg_list[0][which][2]] , label = 'uni_error_rand')
#plt.plot(base_ctg_list[0][which][0], [item[3] for item in base_ctg_list[0][which][2]], label = 'year_error_rand')

plt.legend()
plt.show()

# plt.plot(av1, av2, label = 'uni_error')
# plt.plot(av1, av3, label = 'year_error')
#plt.plot(av1_ran, av2_rand, label = 'uni_error_rand', linestyle='dashed')
#plt.plot(av1_ran, av3_rand, label = 'year_error_rand', linestyle='dashed')
plt.legend()
plt.show()


#******************************

print("**********************")
print("average on baseline: closest to the gold standards")
print("***********")
print("university exact accuracy:: ", mean( [ mean([item[1][-1][0] for item in base_ctg_list[i]])for i in range(num_iteration) ] ) )
print("year exact accuracy:: ", mean( [ mean([item[1][-1][1] for item in base_ctg_list[i]])for i in range(num_iteration) ] ) )

print("**********************")

print("average on baseline: majority extraction method")
print("university exact accuracy:: ", mean( [ mean([item[0] for item in base_ma_list[i]]) for i in range(num_iteration) ] ) )
print("year exact accuracy:: ", mean( [ mean([item[1] for item in base_ma_list[i]]) for i in range(num_iteration) ] ) )

print("**********************")
print('number of total used snippets')
print( mean( [mean(percentage_used_snippets[i]) for i in range(num_iteration)] ) )

print("**********************")
print('average number of changed queries from 7 queries')
print( mean( [mean(num_changed_queries[i]) for i in range(num_iteration)] ) )

"generate a graph indicating which percentage of snippets, change queries each user utilises in the test process."
average_percentage_used_snippets =  [sum(col) / float(len(col)) for col in zip(*percentage_used_snippets)]
average_num_changed_queries =  [sum(col) / float(len(col)) for col in zip(*num_changed_queries)]

x= used_users
plt.scatter(x, average_percentage_used_snippets, label = 'used snippets')
plt.scatter(x, [i/7.0 for i in average_num_changed_queries] , label = 'changed queries')
#plt.grid()
plt.legend()
plt.show()

####average on rewards###
average_reward_aver =  [sum(col) / float(len(col)) for col in zip(*[average_reward(reward_matrix[t]) for t in range(num_iteration)] )]
plt.plot(range(len(average_reward_aver)), average_reward_aver, label = 'average reward evolution')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('rewards')
plt.show()

_ave_university_ave = []
_ave_year_ave = []

for i in range(num_iteration):
    tempo = average_accuracy(accuracy_matrix[i])
    _ave_university_ave.append(tempo[0])
    _ave_year_ave.append(tempo[1])

ave_university_ave  =  [sum(col) / float(len(col)) for col in zip(*_ave_university_ave )]
ave_year_ave  =  [sum(col) / float(len(col)) for col in zip(*_ave_year_ave )]

plt.plot(range(len(ave_university_ave)), ave_university_ave, label = 'university accuracy')
plt.plot(range(len(ave_year_ave)), ave_year_ave, label = 'year accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

# 2 and 40 for db_v1 for db_v1
# 13, 31 and 66 for db_v1 for db_v2

#print(trajectories[0])

#print(accuracy_matrix[0][66])
#print(trajectories[0][66])
#print('***********')
#print(gold_standards[66])
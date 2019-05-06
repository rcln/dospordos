import operator
import os
import pickle
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

name = "DQN_0_db_v2_s_new"
num_iteration = 5

measure_results_matrix = []
reward_matrix = []
base_ctg_list = []
base_ma_list = []
accuracy_matrix = []
final_queries = []
num_changed_queries = []
percentage_used_snippets = []
trajectories = []

"because we have 5 iterations"
for k in range(num_iteration):
    print(k)

    measure_results_matrix.append(pickle.load( open( "../DATA/" + name + "_mrm_" + str(k) + ".pkl", "rb" ) ))
    reward_matrix.append(pickle.load( open( "../DATA/" + name + "_rm_" + str(k) + ".pkl", "rb")))
    base_ctg_list.append(pickle.load( open( "../DATA/" + name + "_ctg_" + str(k) + ".pkl", "rb" ) ))
    base_ma_list.append(pickle.load( open("../DATA/" + name + "_ma_" + str(k) + ".pkl", "rb" ) ))
    accuracy_matrix.append(pickle.load( open( "../DATA/" + name + "_acc_" + str(k) + ".pkl", "rb" ) ))

    final_queries.append(pickle.load(open('../DATA/' + name + '_queries_' + str(k) + '.pkl', 'rb')))
    # print('final rested queries************')
    # print(final_queries)
    # print(len(final_queries))

    num_changed_queries.append(pickle.load(open('../DATA/' + name + '_nc_queries_' + str(k) + '.pkl', 'rb')))
    # print('number of changed queries**********')
    # print(num_changed_queries)
    # print(len(num_changed_queries))

    percentage_used_snippets.append(pickle.load(open('../DATA/' + name + '_per_snippets_' + str(k) + '.pkl', 'rb')))
    # print('percentage of used snippets************')
    # print(percentage_used_snippets)
    # print(len(percentage_used_snippets))

    trajectories.append(pickle.load(open('../DATA/' + name + '_trajectories_' + str(k) + '.pkl', 'rb')))
    # print('trajectories***********')
    # print(trajectories)
    # print(len(trajectories))


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

print("**********************")
print("average on baseline: closest to the gold standards")
print("university exact accuracy:: ", mean( [ mean([item[0] for item in base_ctg_list[i]])for i in range(num_iteration) ] ) )
print("year exact accuracy:: ", mean( [ mean([item[1] for item in base_ctg_list[i]])for i in range(num_iteration) ] ) )

print("**********************")
print("average on baseline: majority extraction method")
print("university exact accuracy:: ", mean( [ mean([item[0] for item in base_ma_list[i]]) for i in range(num_iteration) ] ) )
print("year exact accuracy:: ", mean( [ mean([item[1] for item in base_ma_list[i]]) for i in range(num_iteration) ] ) )

print("**********************")
print('percentage of total used snippets')
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

"AVERAGE ON REWARDS EVOLUTIONS FOR USERS OF THE TEST SET"
def average_reward(_matrix):
    max_len = max([len(item) for item in _matrix])
    average_ = []
    for iter in range(max_len):

        counter = 0
        summ = 0

        for item in _matrix:
            if len(item) > iter:
                counter += 1
                summ += item[iter]

        average_.append(summ/counter)

    return average_

def average_accuracy(_matrix):
    max_len = max([len(item) for item in _matrix])
    average_uni = []
    average_year= []
    for iter in range(max_len):

        counter = (0, 0)
        summ = (0,0)

        for item in _matrix:
            if len(item) > iter:
                counter = tuple(map(operator.add, counter, (1, 1)))
                summ = tuple(map(operator.add, summ, item[iter]))

        average_uni.append(summ[0]/counter[0])
        average_year.append(summ[1]/counter[1])

    return (average_uni, average_year)

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

plt.plot(range(len(ave_university_ave)), ave_university_ave, label = 'average university accuracy')
plt.plot(range(len(ave_year_ave)), ave_year_ave, label = 'average year accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

# 2 and 40 for db_v1 for db_v1
# 13, 31 and 66 for db_v1 for db_v2

print(accuracy_matrix[0][66])
print(trajectories[0][66])
print('***********')
print(gold_standards[66])
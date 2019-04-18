import operator
import os
import pickle
from itertools import count

import matplotlib.pyplot as plt

from numpy import mean

name = "DQN_0_db_v1_ns_new"

measure_results_matrix = pickle.load( open( "../DATA/" + name + "_mrm.pkl", "rb" ) )
# print('different measures for each user as Pu, Ru, Fu, Py, Ry, Fy*************')
# print(measure_results_matrix)
# print(len(measure_results_matrix))

reward_matrix = pickle.load( open( "../DATA/" + name + "_rm.pkl", "rb" ) )
# print("accumulated rewards per epoch***********")
# print(reward_matrix)
# print(len(reward_matrix))

base_ctg_list = pickle.load( open( "../DATA/" + name + "_ctg.pkl", "rb" ) )
# print("baseline: closest to the gold standards************")
# print(base_ctg_list)
# print(len(base_ctg_list))

base_ma_list = pickle.load( open( "../DATA/" + name + "_ma.pkl", "rb" ) )
# print("baseline: majority aggregation method***************")
# print(base_ma_list)
# print(len(base_ma_list))

accuracy_matrix = pickle.load( open( "../DATA/" + name + "_acc.pkl", "rb" ) )
# print("total accuracy of our method. exact accuracy*************")
# print(accuracy_matrix)
# print(len(accuracy_matrix))

used_users = pickle.load(open('../DATA/' + name + '_uu.pkl', 'rb'))
# print('list of used users******')
# print(used_users)
# print(len(used_users))

final_queries = pickle.load(open('../DATA/' + name + '_queries.pkl', 'rb'))
# print('final rested queries************')
# print(final_queries)
# print(len(final_queries))

num_changed_queries = pickle.load(open('../DATA/' + name + '_nc_queries.pkl', 'rb'))
# print('number of changed queries**********')
# print(num_changed_queries)
# print(len(num_changed_queries))

percentage_used_snippets =  pickle.load(open('../DATA/' + name + '_per_snippets.pkl', 'rb'))
# print('percentage of used snippets************')
# print(percentage_used_snippets)
# print(len(percentage_used_snippets))

gold_standards = pickle.load( open('../DATA/' + name + '_gold_standards.pkl', 'rb'))
print('gold standards*********')
print(gold_standards)
print(len(gold_standards))

trajectories = pickle.load(open('../DATA/' + name + '_trajectories.pkl', 'rb'))
print('trajectories***********')
print(trajectories)
print(len(trajectories))




print("**********************")
print("average on total exact accuracy")
print("university exact accuracy:: ", mean([item[-1][0] for item in accuracy_matrix]))
print("year exact accuracy:: ", mean([item[-1][1] for item in accuracy_matrix]))

print("**********************")
print("average precision, recall, F1 score")
print("Pu, Ru, Fu, Py, Ry, Fy")
print("university precision:: ", mean([item[0][0] for item in measure_results_matrix]))
print("university recall:: ", mean([item[0][1] for item in measure_results_matrix]))
print("university F1 score:: ", mean([item[0][2] for item in measure_results_matrix]))

print("year precision:: ", mean([item[0][3] for item in measure_results_matrix]))
print("year recall:: ", mean([item[0][4] for item in measure_results_matrix]))
print("year F1 score:: ", mean([item[0][5] for item in measure_results_matrix]))

print("**********************")
print("average on baseline: closest to the gold standards")
print("university exact accuracy:: ", mean([item[0] for item in base_ctg_list]))
print("year exact accuracy:: ", mean([item[1] for item in base_ctg_list]))

print("**********************")
print("average on baseline: majority extraction method")
print("university exact accuracy:: ", mean([item[0] for item in base_ma_list]))
print("year exact accuracy:: ", mean([item[1] for item in base_ma_list]))

print("**********************")
print('percentage of total used snippets')
print(mean(percentage_used_snippets))

print("**********************")
print('average number of changed queries from 7 queries')
print(mean(num_changed_queries))

"generate a graph indicating which percentage of snippets, change queries each user utilises in the test process."
#x= used_users
#plt.scatter(x, percentage_used_snippets, label = 'used snippets')
#plt.scatter(x, [i/7.0 for i in num_changed_queries], label = 'changed queries')
#plt.grid()
#plt.legend()
#plt.show()

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

    return  average_


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

average_reward = average_reward(reward_matrix)
plt.plot(range(len(average_reward)), average_reward, label = 'average reward evolution')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('rewards')
plt.show()


(ave_university, ave_year) = average_accuracy(accuracy_matrix)
plt.plot(range(len(ave_university)), ave_university, label = 'average university accuracy')
plt.plot(range(len(ave_year)), ave_year, label = 'average year accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

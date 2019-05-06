import operator
import pickle
import matplotlib.pyplot as plt


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


reward_matrix_0_v2_ns = []
reward_matrix_1_v2_ns = []
reward_matrix_0_v2_s = []
reward_matrix_1_v2_s = []

reward_matrix_0_v1_ns = []
reward_matrix_1_v1_ns = []
reward_matrix_0_v1_s = []
reward_matrix_1_v1_s = []


accuracy_matrix_0_v2_ns = []
accuracy_matrix_1_v2_ns = []
accuracy_matrix_0_v2_s = []
accuracy_matrix_1_v2_s = []

accuracy_matrix_0_v1_ns = []
accuracy_matrix_1_v1_ns = []
accuracy_matrix_0_v1_s = []
accuracy_matrix_1_v1_s = []

num_iteration = 5
for k in range(num_iteration):
    reward_matrix_0_v2_ns.append(pickle.load(open("../DATA/" + "DQN_0_db_v2_ns_new" + "_rm_" + str(k) + ".pkl", "rb")))
    reward_matrix_1_v2_ns.append(pickle.load(open("../DATA/" + "DQN_1_db_v2_ns_new" + "_rm_" + str(k) + ".pkl", "rb")))
    reward_matrix_0_v2_s.append(pickle.load(open("../DATA/" + "DQN_0_db_v2_s_new" + "_rm_" + str(k) + ".pkl", "rb")))
    reward_matrix_1_v2_s.append(pickle.load(open("../DATA/" + "DQN_1_db_v2_s_new" + "_rm_" + str(k) + ".pkl", "rb")))

    reward_matrix_0_v1_ns.append(pickle.load(open("../DATA/" + "DQN_0_db_v1_ns_new" + "_rm_" + str(k) + ".pkl", "rb")))
    reward_matrix_1_v1_ns.append(pickle.load(open("../DATA/" + "DQN_1_db_v1_ns_new" + "_rm_" + str(k) + ".pkl", "rb")))
    reward_matrix_0_v1_s.append(pickle.load(open("../DATA/" + "DQN_0_db_v1_s_new" + "_rm_" + str(k) + ".pkl", "rb")))
    reward_matrix_1_v1_s.append(pickle.load(open("../DATA/" + "DQN_1_db_v1_s_new" + "_rm_" + str(k) + ".pkl", "rb")))



    accuracy_matrix_0_v2_ns.append(pickle.load(open("../DATA/" + "DQN_0_db_v2_ns_new" + "_acc_" + str(k) + ".pkl", "rb")))
    accuracy_matrix_1_v2_ns.append(pickle.load(open("../DATA/" + "DQN_1_db_v2_ns_new" + "_acc_" + str(k) + ".pkl", "rb")))
    accuracy_matrix_0_v2_s.append(pickle.load(open("../DATA/" + "DQN_0_db_v2_s_new" + "_acc_" + str(k) + ".pkl", "rb")))
    accuracy_matrix_1_v2_s.append(pickle.load(open("../DATA/" + "DQN_1_db_v2_s_new" + "_acc_" + str(k) + ".pkl", "rb")))

    accuracy_matrix_0_v1_ns.append(pickle.load(open("../DATA/" + "DQN_0_db_v1_ns_new" + "_acc_" + str(k) + ".pkl", "rb")))
    accuracy_matrix_1_v1_ns.append(pickle.load(open("../DATA/" + "DQN_1_db_v1_ns_new" + "_acc_" + str(k) + ".pkl", "rb")))
    accuracy_matrix_0_v1_s.append(pickle.load(open("../DATA/" + "DQN_0_db_v1_s_new" + "_acc_" + str(k) + ".pkl", "rb")))
    accuracy_matrix_1_v1_s.append(pickle.load(open("../DATA/" + "DQN_1_db_v1_s_new" + "_acc_" + str(k) + ".pkl", "rb")))



average_reward_aver_0_v2_ns =  [sum(col) / float(len(col)) for col in zip(*[average_reward(reward_matrix_0_v2_ns[t]) for t in range(num_iteration)] )]
average_reward_aver_1_v2_ns =  [sum(col) / float(len(col)) for col in zip(*[average_reward(reward_matrix_1_v2_ns[t]) for t in range(num_iteration)] )]
average_reward_aver_0_v2_s =  [sum(col) / float(len(col)) for col in zip(*[average_reward(reward_matrix_0_v2_s[t]) for t in range(num_iteration)] )]
average_reward_aver_1_v2_s =  [sum(col) / float(len(col)) for col in zip(*[average_reward(reward_matrix_1_v2_s[t]) for t in range(num_iteration)] )]
average_reward_aver_0_v1_ns =  [sum(col) / float(len(col)) for col in zip(*[average_reward(reward_matrix_0_v1_ns[t]) for t in range(num_iteration)] )]
average_reward_aver_1_v1_ns =  [sum(col) / float(len(col)) for col in zip(*[average_reward(reward_matrix_1_v1_ns[t]) for t in range(num_iteration)] )]
average_reward_aver_0_v1_s =  [sum(col) / float(len(col)) for col in zip(*[average_reward(reward_matrix_0_v1_s[t]) for t in range(num_iteration)] )]
average_reward_aver_1_v1_s =  [sum(col) / float(len(col)) for col in zip(*[average_reward(reward_matrix_1_v1_s[t]) for t in range(num_iteration)] )]

#plt.plot(range(len(average_reward_aver_0_v2_ns)), average_reward_aver_0_v2_ns, label = 'DQN + db_v2_ns')
plt.plot(range(len(average_reward_aver_1_v2_ns)), average_reward_aver_1_v2_ns, label = 'DQN + RE + db_v2_ns')
plt.plot(range(len(average_reward_aver_0_v2_s)), average_reward_aver_0_v2_s, label = 'DQN + db_v2_s')
plt.plot(range(len(average_reward_aver_1_v2_s )), average_reward_aver_1_v2_s , label = 'DQN + RE + db_v2_s')
plt.plot(range(len(average_reward_aver_0_v1_ns )), average_reward_aver_0_v1_ns , label = 'DQN + db_v1_ns')
plt.plot(range(len(average_reward_aver_1_v1_ns )), average_reward_aver_1_v1_ns , label = 'DQN + RE + db_v1_ns')
plt.plot(range(len(average_reward_aver_0_v1_s)), average_reward_aver_0_v1_s, label = 'DQN + db_v1_s')
plt.plot(range(len(average_reward_aver_1_v1_s)), average_reward_aver_1_v1_s, label = 'DQN + RE + db_v1_s')

plt.legend()
plt.xlabel('epochs')
plt.ylabel('rewards')
plt.show()



_ave_university_ave_0_v2_ns = []
_ave_year_ave_0_v2_ns = []

_ave_university_ave_1_v2_ns = []
_ave_year_ave_1_v2_ns = []

_ave_university_ave_0_v2_s = []
_ave_year_ave_0_v2_s = []

_ave_university_ave_1_v2_s = []
_ave_year_ave_1_v2_s = []

_ave_university_ave_0_v1_ns = []
_ave_year_ave_0_v1_ns = []

_ave_university_ave_1_v1_ns = []
_ave_year_ave_1_v1_ns = []

_ave_university_ave_0_v1_s = []
_ave_year_ave_0_v1_s = []

_ave_university_ave_1_v1_s = []
_ave_year_ave_1_v1_s = []

for i in range(num_iteration):
    tempo = average_accuracy(accuracy_matrix_0_v2_ns[i])
    _ave_university_ave_0_v2_ns.append(tempo[0])
    _ave_year_ave_0_v2_ns.append(tempo[1])

    tempo = average_accuracy(accuracy_matrix_1_v2_ns[i])
    _ave_university_ave_1_v2_ns.append(tempo[0])
    _ave_year_ave_1_v2_ns.append(tempo[1])

    tempo = average_accuracy(accuracy_matrix_0_v2_s[i])
    _ave_university_ave_0_v2_s.append(tempo[0])
    _ave_year_ave_0_v2_s.append(tempo[1])

    tempo = average_accuracy(accuracy_matrix_1_v2_s[i])
    _ave_university_ave_1_v2_s.append(tempo[0])
    _ave_year_ave_1_v2_s.append(tempo[1])

    tempo = average_accuracy(accuracy_matrix_0_v1_ns[i])
    _ave_university_ave_0_v1_ns.append(tempo[0])
    _ave_year_ave_0_v1_ns.append(tempo[1])

    tempo = average_accuracy(accuracy_matrix_1_v1_ns[i])
    _ave_university_ave_1_v1_ns.append(tempo[0])
    _ave_year_ave_1_v1_ns.append(tempo[1])

    tempo = average_accuracy(accuracy_matrix_0_v1_s[i])
    _ave_university_ave_0_v1_s.append(tempo[0])
    _ave_year_ave_0_v1_s.append(tempo[1])

    tempo = average_accuracy(accuracy_matrix_1_v1_s[i])
    _ave_university_ave_1_v1_s.append(tempo[0])
    _ave_year_ave_1_v1_s.append(tempo[1])


ave_university_ave_0_v2_ns   =  [sum(col) / float(len(col)) for col in zip(*_ave_university_ave_0_v2_ns  )]
ave_year_ave_0_v2_ns   =  [sum(col) / float(len(col)) for col in zip(*_ave_year_ave_0_v2_ns  )]

ave_university_ave_1_v2_ns   =  [sum(col) / float(len(col)) for col in zip(*_ave_university_ave_1_v2_ns  )]
ave_year_ave_1_v2_ns   =  [sum(col) / float(len(col)) for col in zip(*_ave_year_ave_1_v2_ns  )]

ave_university_ave_0_v2_s   =  [sum(col) / float(len(col)) for col in zip(*_ave_university_ave_0_v2_s  )]
ave_year_ave_0_v2_s   =  [sum(col) / float(len(col)) for col in zip(*_ave_year_ave_0_v2_s  )]

ave_university_ave_1_v2_s   =  [sum(col) / float(len(col)) for col in zip(*_ave_university_ave_1_v2_s  )]
ave_year_ave_1_v2_s   =  [sum(col) / float(len(col)) for col in zip(*_ave_year_ave_1_v2_s  )]

ave_university_ave_0_v1_ns   =  [sum(col) / float(len(col)) for col in zip(*_ave_university_ave_0_v1_ns  )]
ave_year_ave_0_v1_ns   =  [sum(col) / float(len(col)) for col in zip(*_ave_year_ave_0_v1_ns  )]

ave_university_ave_1_v1_ns   =  [sum(col) / float(len(col)) for col in zip(*_ave_university_ave_1_v1_ns  )]
ave_year_ave_1_v1_ns   =  [sum(col) / float(len(col)) for col in zip(*_ave_year_ave_1_v1_ns  )]

ave_university_ave_0_v1_s   =  [sum(col) / float(len(col)) for col in zip(*_ave_university_ave_0_v1_s  )]
ave_year_ave_0_v1_s   =  [sum(col) / float(len(col)) for col in zip(*_ave_year_ave_0_v1_s  )]

ave_university_ave_1_v1_s   =  [sum(col) / float(len(col)) for col in zip(*_ave_university_ave_1_v1_s  )]
ave_year_ave_1_v1_s   =  [sum(col) / float(len(col)) for col in zip(*_ave_year_ave_1_v1_s  )]


#plt.plot(range(len(ave_university_ave_0_v2_ns)), ave_university_ave_0_v2_ns, label = 'university + DQN + db_v2_ns')
#plt.plot(range(len(ave_year_ave_0_v2_ns )), ave_year_ave_0_v2_ns , label = 'year + DQN + db_v2_ns')
plt.plot(range(len(ave_university_ave_1_v2_ns)), ave_university_ave_1_v2_ns, label = 'university + DQN + RE + db_v2_ns')
#plt.plot(range(len(ave_year_ave_1_v2_ns )), ave_year_ave_1_v2_ns , label = 'year + DQN + RE + db_v2_ns')
plt.plot(range(len(ave_university_ave_0_v2_s)), ave_university_ave_0_v2_s, label = 'university + db_v2_s')
#plt.plot(range(len(ave_year_ave_0_v2_s )), ave_year_ave_0_v2_s , label = 'year + db_v2_s')
plt.plot(range(len(ave_university_ave_1_v2_s)), ave_university_ave_1_v2_s, label = 'university + DQN + RE + db_v2_s')
#plt.plot(range(len(ave_year_ave_1_v2_s )), ave_year_ave_1_v2_s , label = 'year + DQN + RE + db_v2_s')

plt.plot(range(len(ave_university_ave_0_v1_ns)), ave_university_ave_0_v1_ns, label = 'university + DQN + db_v1_ns')
#plt.plot(range(len(ave_year_ave_0_v1_ns )), ave_year_ave_0_v1_ns , label = 'year + DQN + db_v1_ns')
plt.plot(range(len(ave_university_ave_1_v1_ns)), ave_university_ave_1_v1_ns, label = 'university + DQN + RE + db_v1_ns')
#plt.plot(range(len(ave_year_ave_1_v1_ns )), ave_year_ave_1_v1_ns , label = 'year + DQN + RE + db_v1_ns')
plt.plot(range(len(ave_university_ave_0_v1_s)), ave_university_ave_0_v1_s, label = 'university + db_v1_s')
#plt.plot(range(len(ave_year_ave_0_v1_s )), ave_year_ave_0_v1_s , label = 'year + db_v1_s')
plt.plot(range(len(ave_university_ave_1_v1_s)), ave_university_ave_1_v1_s, label = 'university + DQN + RE + db_v1_s')
#plt.plot(range(len(ave_year_ave_1_v1_s )), ave_year_ave_1_v1_s , label = 'university + DQN + RE + db_v1_s')

plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()



#plt.plot(range(len(ave_year_ave_0_v2_ns )), ave_year_ave_0_v2_ns , label = 'year + DQN + db_v2_ns')
plt.plot(range(len(ave_year_ave_1_v2_ns )), ave_year_ave_1_v2_ns , label = 'year + DQN + RE + db_v2_ns')
plt.plot(range(len(ave_year_ave_0_v2_s )), ave_year_ave_0_v2_s , label = 'year + db_v2_s')
plt.plot(range(len(ave_year_ave_1_v2_s )), ave_year_ave_1_v2_s , label = 'year + DQN + RE + db_v2_s')
plt.plot(range(len(ave_year_ave_0_v1_ns )), ave_year_ave_0_v1_ns , label = 'year + DQN + db_v1_ns')
plt.plot(range(len(ave_year_ave_1_v1_ns )), ave_year_ave_1_v1_ns , label = 'year + DQN + RE + db_v1_ns')
plt.plot(range(len(ave_year_ave_0_v1_s )), ave_year_ave_0_v1_s , label = 'year + db_v1_s')
plt.plot(range(len(ave_year_ave_1_v1_s )), ave_year_ave_1_v1_s , label = 'year + DQN + RE + db_v1_s')

plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
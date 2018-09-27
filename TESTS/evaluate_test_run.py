import pickle
import numpy as np
import pandas as pd
import argparse

import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser("evaluate_test_run")
    parser.add_argument("-r", help="Aumulative reward pkl name")
    parser.add_argument("-mr", help="Measurement results pkl name")
    parser.add_argument("-acc", help="Accuracy pkl name")
    parser.add_argument("-g", help="Do you want graphs?", required=False,  default=0)
    args = parser.parse_args()

    if args.r:
        rm_objects = []
        with (open("../DATA/" + args.r, "rb")) as openfile:
            while True:
                try:
                    rm_objects = pickle.load(openfile)
                except EOFError:
                    break
        reward_matrix = np.zeros([len(rm_objects), len(max(rm_objects, key=lambda x: len(x)))])
        reward_matrix[:] = None
        for i, j in enumerate(rm_objects):
            reward_matrix[i][0:len(j)] = j
        reward_matrix = pd.DataFrame(reward_matrix)
        reward_matrix_r = []
        for i in range(0, reward_matrix.shape[1]):
            tmp_ar = [x for x in reward_matrix[i] if not np.isnan(x)]
            reward_matrix_r.append(sum(tmp_ar) / len(tmp_ar))

        print("Avg of cumulative reward:", reward_matrix_r)

        if args.g:
            plt.plot(reward_matrix_r)
            plt.ylabel('cumulative reward')
            plt.xlabel('epocs')
            plt.show()

    if args.acc:
        acc_objects = []
        with (open("../DATA/DQN_0_db_v1_ns_acc.pkl", "rb")) as openfile:
            while True:
                try:
                    acc_objects = pickle.load(openfile)
                except EOFError:
                    break

        accuracy_uni = np.zeros([len(acc_objects), len(max(acc_objects, key=lambda x: len(x)))])
        accuracy_years = np.zeros([len(acc_objects), len(max(acc_objects, key=lambda x: len(x)))])

        accuracy_uni[:] = None
        accuracy_years[:] = None

        for i, j in enumerate(acc_objects):
            accuracy_uni[i][0:len(j)] = [x[0] for x in j]
            accuracy_years[i][0:len(j)] = [x[1] for x in j]

        accuracy_uni = pd.DataFrame(accuracy_uni)
        accuracy_years = pd.DataFrame(accuracy_years)

        accuracy_uni_r = []
        accuracy_years_r = []

        for i in range(0, accuracy_uni.shape[1]):
            tmp_ar = [x for x in accuracy_uni[i] if not np.isnan(x)]
            accuracy_uni_r.append(sum(tmp_ar) / len(tmp_ar))
            tmp_ar = [x for x in accuracy_years[i] if not np.isnan(x)]
            accuracy_years_r.append(sum(tmp_ar) / len(tmp_ar))

        print("Avg of University accuracy:", accuracy_uni_r)
        print("Avg of Years accuracy:", accuracy_years_r)

        if args.g:
            plt.plot(accuracy_uni_r)
            plt.ylabel('accuracy uni')
            plt.xlabel('epocs')
            plt.show()

            plt.plot(accuracy_years_r)
            plt.ylabel('accuracy years')
            plt.xlabel('epocs')
            plt.show()
# mrm_objects = []
# with (open("../DATA/" + args.r, "rb")) as openfile:
#     while True:
#         try:
#             mrm_objects = pickle.load(openfile)
#         except EOFError:
#             break

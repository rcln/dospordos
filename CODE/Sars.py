# -*- coding: utf-8 -*-
import numpy as np


class Sars:
    def __init__(self, s=None, a=None, r=None, s_prime=None, random=False):
        if random:
            self.s, self.a, self.r, self.s_prime = self.get_random_sars()
        else:
            self.s = s
            self.a = a
            self.r = np.array(r)
            self.s_prime = s_prime

    @staticmethod
    def get_random_action_vector(size: int):
        action_vector = np.zeros([1, size])
        action_vector[0, np.random.randint(0, size)] = 1
        return action_vector

    def get_random_sars(self):
        A = np.random.uniform(0.0, 10.0)
        B = np.random.uniform(0.0, 10.0)
        a = self.get_random_action_vector(6)
        r = np.array([np.random.uniform(-20.0, 0.0)])
        s = np.concatenate((self.get_random_action_vector(7),
                            np.array([np.random.uniform(0.0, 1.0)]),
                            self.get_random_action_vector(4),
                            np.array([np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0),
                                      np.random.uniform(0.0, 1.0)]),
                            np.array((A, B, A + B)),
                            np.array([np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)]),
                            np.array([np.random.randint(0, 2)]),
                            np.zeros(shape=(27386, ))))
        A = np.random.uniform(0.0, 10.0)
        B = np.random.uniform(0.0, 10.0)
        s_prime = np.concatenate((self.get_random_action_vector(7),
                                  np.array([np.random.uniform(0.0, 1.0)]),
                                  self.get_random_action_vector(4),
                                  np.array([np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0),
                                            np.random.uniform(0.0, 1.0)]),
                                  np.array((A, B, A + B)),
                                  np.array([np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)]),
                                  np.array([np.random.randint(0, 2)]),
                                  np.zeros(shape=(27386, ))))
        return s, a, r, s_prime


if __name__ == "__main__":
    sars1 = Sars()
    print("State: ", sars1.s)
    print("Action: ", sars1.a)
    print("Reward: ", sars1.r)
    print("Previous state: ", sars1.s_prime)

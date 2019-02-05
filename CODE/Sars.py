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

    @staticmethod
    def get_random_action_vector_pa(size: int):
        action_vector = [0] * size
        action_vector[np.random.randint(0, size)] = 1
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
                                  np.array([np.random.randint(0, 2)])
                                  , np.zeros(shape=(27386, ))
                                  ))

        #TODO : PA why state has dimension 19(+27386) instead of dimension 21(+27386)?!!!
        # TODO: Answer. The dimension is 21, the missing numbers are in the line  np.array((A, B, A + B)),
        # However, the function is deprecated, needs to be updated because the shape of get_random_action_vector
        # changed from (7,)  to (1,7)

        A = np.random.uniform(0.0, 10.0)
        B = np.random.uniform(0.0, 10.0)
        s_prime = np.concatenate((self.get_random_action_vector(7),
                                  np.array([np.random.uniform(0.0, 1.0)]),
                                  self.get_random_action_vector(4),
                                  np.array([np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0),
                                            np.random.uniform(0.0, 1.0)]),
                                  np.array((A, B, A + B)),
                                  np.array([np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)]),
                                  np.array([np.random.randint(0, 2)])
                                  , np.zeros(shape=(27386, ))
                                  ))
        return s, a, r, s_prime

if __name__ == "__main__":
    sars1 = Sars(random= False)
    print("random action ", sars1.get_random_action_vector_pa(6))


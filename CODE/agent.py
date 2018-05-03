# -*- coding: utf-8 -*-
import numpy as np
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import plot_model


class Agent:

    def __init__(self, env, shape=(27,)):
        self.env = env
        self.network = Network(shape)

    # grid functions
    def next_snippet(self):

        try:
            if self.env.queues[self.env.current_queue].qsize() == 0:
                self.env.current_data = {"number_snippet": "1000000", "text": "", "cite": "", "search": "", "title": "",
                                         "engine_search": "-1", "id_person": ""}
            else:
                self.env.current_data = self.env.queues[self.env.current_queue].get(False)
        except KeyError:
            print("ERROR in next_snippet\n current queue: ", self.env.current_queue)
            print("Queues ", self.env.queues)
            print("DATA ", self.env.current_data)


    def change_queue(self, queue=None):
        if queue is None:
            tmp = self.env.current_queue + 1
            if tmp >= len(self.env.queues):
                self.env.current_queue = 0
            else:
                self.env.current_queue = tmp
        else:
            if queue >= len(self.env.queues):
                self.env.current_queue = 0
            else:
                self.env.current_queue = queue

    # db functions
    def delete_current_db(self, i=None):
        if len(self.env.current_db) > 0:
            self.env.current_db.pop()

    def add_current_db(self, i=None):
        self.env.current_db.append(tuple(self.env.info_snippet))

    @staticmethod
    def keep_current_db():
        pass

    def actions_to_take(self, action_activation_vector):
        num_l = np.nonzero(action_activation_vector[0])
        num = num_l[0][0]
        actions_db = (self.delete_current_db, self.add_current_db, self.keep_current_db)
        actions_grid = (self.next_snippet, self.change_queue)
        return actions_grid[int(num/3)], actions_db[num % 3]

    def print_model(self):
        plot_model(self.network, to_file='model.png')


class Network:

    def __init__(self, tensor_shape):
        self.model = self._create_model(tensor_shape)
        self.model.compile(loss=keras.losses.mean_squared_error,
                           optimizer=keras.optimizers.adam(),
                           metrics=['accuracy'])
        print(self.model.summary())

    @staticmethod
    def _create_model(tensor_shape):
        x = Input(shape=tensor_shape)
        h = Dense(units=10, activation='relu')(x)
        h = Dropout(0.25)(h)
        h = Dense(units=10, activation='relu')(h)
        h = Dropout(0.25)(h)
        o = Dense(units=1, activation='linear')(h)
        return Model(inputs=x, outputs=o)

    def fit(self, x_train, y_train, epochs, batch_size):
        self.model.fit(x_train, y_train, # batch_size=batch_size,
                       epochs=epochs,
                       verbose=1)

    def fit_generator(self, gen, steps_per_epoch, epochs):
        self.model.fit_generator(generator=gen,
                                 # steps_per_epoch=steps_per_epoch,
                                 # epochs=epochs,
                                 verbose=0)

    # Desc: (loss, accuracy)
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)

    def save_weights(self, path):
        self.model.save_weights(filepath=path)

    def load_weights(self, path):
        self.model.load_weights(filepath=path, by_name=False)

    def predict(self, x):
        return self.model.predict(x)



# -*- coding: utf-8 -*-

# Todo implement the actions
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout


class Agent:

    def __init__(self, env):
        self.env = env

    def next_snippet(self):
        self.env.current_data = self.env.queues[self.env.current_queue].get()

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

    def delete_current_db(self,i=None):
        self.env.current_db.pop()

    def add_current_db(self,i=None): #
        self.env.current_db.append(self.env.info_snippet)

    @staticmethod
    def keep_current_db(): #
        pass

    #TODO  theo if de 6 casos todas las combinaciones
    def actions_to_take(self,action_activation_vector):
        #example
        #return self.delete_current_db, self.next_snippet
        pass


class Network:

    def __init__(self, tensor_shape):
        self.model = self._create_model(tensor_shape)
        self.model.compile(loss=keras.losses.mean_squared_error,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

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
        self.model.fit(x_train, y_train, batch_size=batch_size,
                       epochs=epochs,
                       verbose=0)

    def fit_generator(self, gen, steps_per_epoch, epochs):
        self.model.fit_generator(generator=gen,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 verbose=0)

    # (loss, accuracy)
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)

    def save_weights(self, path):
        self.model.save_weights(filepath=path)

    def load_weights(self, path):
        self.model.load_weights(filepath=path, by_name=False)

    def predict(self, x):
        return self.model.predict(x)



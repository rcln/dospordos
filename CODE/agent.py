# -*- coding: utf-8 -*-
import numpy as np
import keras
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.utils import plot_model


class Agent:

    def __init__(self, env, shape=(26,)):
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

    def next_snippet_pa(self):

        try:
            if self.env.env_core.queues[self.env.env_core.current_queue].qsize() == 0:
                # TODO PA: I should think, what to do if a queue of a query for a given person_id is empty, should we stop the whole
                # process or use another query randomly, but this means choose an action as a query obligatory!!
                while True:
                    self.change_queue()
                    if self.env.env_core.queues[self.env.env_core.current_queue].qsize() != 0:
                        self.env.env_core.que_changed_obligatory = True
                        break
                    else:
                        print("*** all snippents from all queries have been searched ****")
                        break
                        #raise NameError('***QUEUE IS EMPTY***')

            else:
                # this is equal to popping a snippet from the current query
                self.env.env_core.current_data = self.env.env_core.queues[self.env.env_core.current_queue].get(False)
        except KeyError:
            print("ERROR in next_snippet\n current queue: ", self.env.env_core.current_queue)
            print("Queues ", self.env.env_core.queues)
            print("DATA ", self.env.env_core.current_data)

    def change_queue(self, queue=None):
        if queue is None:
            tmp = self.env.env_core.current_queue + 1
            if tmp >= len(self.env.env_core.queues):
                self.env.env_core.current_queue = 0
            else:
                self.env.env_core.current_queue = tmp
        else:
            if queue >= len(self.env.env_core.queues):
                self.env.env_core.current_queue = 0
            else:
                self.env.env_core.current_queue = queue

    # db functions
    def delete_current_db(self, i=None):
        if len(self.env.env_core.current_db) > 0:
            self.env.env_core.current_db.pop()

    def add_current_db(self, i=None):
        self.env.env_core.current_db.append(self.env.env_core.info_snippet[0])

    @staticmethod
    def keep_current_db():
        pass

    def actions_to_take(self, action_activation_vector):
        num_l = np.nonzero(action_activation_vector[0])
        num = num_l[0][0]
        actions_db = (self.delete_current_db, self.add_current_db, self.keep_current_db)
        actions_grid = (self.next_snippet, self.change_queue)
        return actions_grid[int(num/3)], actions_db[num % 3]

    # ToDo Message to Pegah: I had to comment this just to improve the time execution a little bit. But is the same
    # def actions_to_take_pa(self, action_activation_vector):

        # "if query is called"
        #print(action_activation_vector)
        # if action_activation_vector[-1] == 1:
        #     self.change_queue()
        #if the NER should be reconciled
        # elif action_activation_vector[-1] == 0:

            # if action_activation_vector[0] == 1:
            #     "Organisation is accepted"
            # if action_activation_vector[1] == 1:
            #     "Organization is rejected"
            # if action_activation_vector[2] == 1:
            #     "date is accepted"
            # if action_activation_vector[3] == 1:
            #     "date is rejected"
            # self.next_snippet_pa()

        # if action_activation_vector[-1]:
        #     self.change_queue()
        # else:
        #     self.next_snippet_pa()

        # return action_activation_vector

    def actions_to_take_pa(self, action_activation_vector):
        print("action activation", action_activation_vector[0])
        if action_activation_vector[-1]:
            self.change_queue()
        else:
            self.next_snippet_pa()
        return action_activation_vector

    def print_model(self):
        plot_model(self.network, to_file='model.png')


class Network:

    def __init__(self, tensor_shape):
        self.tensor_shape = tensor_shape
        self.model = self._create_model(tensor_shape)
        self.model.compile(loss=keras.losses.mean_squared_error,
                           optimizer=keras.optimizers.adam(),
                           metrics=['accuracy'])
        # print(self.model.summary())


    #TODO PA: we can play with the NN model. The used model in the MIT paper is: linear, RELU, linear, RELU, linear

    """
    our NN model for predicting Q(s,a):
    
        Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 28)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                280       
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 10)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                110       
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 10)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 11        
    ================================================================= 
    """

    @staticmethod
    def _create_model(tensor_shape):
        x = Input(shape=tensor_shape)
        h = Dense(units=10, activation='relu')(x)
        h = Dropout(0.25)(h)
        h = Dense(units=10, activation='relu')(h)
        h = Dropout(0.25)(h)
        o = Dense(units=1, activation='linear')(h)
        return Model(inputs=x, outputs=o)

    def fit(self, x_train, y_train, epochs, batch_size, callbacks=None):
        if callbacks is None:
            self.model.fit(x_train, y_train,  # batch_size=batch_size,
                           epochs=epochs,
                           verbose=1)  # could be 1
        else:
            self.model.fit(x_train, y_train,  # batch_size=batch_size,
                           epochs=epochs, callbacks=callbacks,
                           verbose=1)  # could be 1

    def fit_generator(self, gen, steps_per_epoch, epochs):
        self.model.fit_generator(generator=gen,
                                 # steps_per_epoch=steps_per_epoch,
                                 # epochs=epochs,
                                 verbose=1)

    # Desc: (loss, accuracy)
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)

    def save_weights(self, path):
        self.model.save_weights(filepath=path)

    def load_weights(self, path):
        self.model.load_weights(filepath=path, by_name=False)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        keras.models.load_model(path)

    def predict(self, x):
        return self.model.predict(x)


class EarlyStopByLossVal(Callback):

    def __init__(self, monitor='loss', value=0.01, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            print("Early stopping requires %s available!" % self.monitor)
            # warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
            return
        if current < self.value:
            self.model.stop_training = True
            print("Epoch %05d: early stopping THR" % epoch)
            with open('tmp_record', 'w') as f:
                f.write("1")


# Wrapper of EarlyStopping callback
def EarlyStopping(monitor='loss', min_delta=0.01, patience=0, verbose=0, mode='auto', baseline=None):
    return keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta,
                                         patience=patience, verbose=verbose, mode=mode, baseline=baseline)

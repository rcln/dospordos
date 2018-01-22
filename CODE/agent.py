# -*- coding: utf-8 -*-

# Todo implement the actions
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout


class Agent:

    def __init__(self, queries, current_query):
        self.queries = queries
        self.current_query = current_query
        self.current_snippet = None

    def next_snippet(self):
        self.current_snippet = self.queries[self.current_query].get()

    def change_query(self, query):
        self.current_query = query



    @staticmethod
    def stop():
        pass

    @staticmethod
    def change_db():
        pass

    @staticmethod
    def update_db():
        pass


# todo duda con el modelo para las 2 salidas, función de perdida
class Model:

    def __init__(self, tensor_shape):
        self.model = self._create_model(tensor_shape)
        self.model.compile(loss=keras.losses.mean_squared_error,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

    @staticmethod
    def _create_model(tensor_shape):
        x = Input(tensor_shape)
        h = Dropout(0.25)(x)
        h = Dense(units=10, activation='relu')(h)
        o = Dense(units=5, activation='softmax')(h)
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



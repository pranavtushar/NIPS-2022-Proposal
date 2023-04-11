from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Layer
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.keras.models import Sequential
from kerashypetune import KerasGridSearch

class T2V(Layer):
    
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)
        
    def build(self, input_shape):

        self.W = self.add_weight(name='W',
                                shape=(input_shape[-1], self.output_dim),
                                initializer='uniform',
                                trainable=True)

        self.P = self.add_weight(name='P',
                                shape=(input_shape[1], self.output_dim),
                                initializer='uniform',
                                trainable=True)

        self.w = self.add_weight(name='w',
                                shape=(input_shape[1], 1),
                                initializer='uniform',
                                trainable=True)

        self.p = self.add_weight(name='p',
                                shape=(input_shape[1], 1),
                                initializer='uniform',
                                trainable=True)

        super(T2V, self).build(input_shape)
        
    def call(self, x):
        
        original = self.w * x + self.p #if i = 0
        sin_trans = K.sin(K.dot(x, self.W) + self.P) # Frequecy and phase shift of sine function, learnable parameters. if 1 <= i <= k
        
        return K.concatenate([sin_trans, original], -1)

class Time2Vec:

    def __init__(self, 
                param={
                    'unit': 32,
                    't2v_dim': 64,
                    'lr': 1e-2, 
                    'act': 'relu', 
                    'epochs': 20,
                    'batch_size': 1024
                }, dim=20):
        self.param = param
        self.dim = dim

    def set_LSTM(self):  
        inp = layers.Input(shape=(self.dim,2))
        x = T2V(self.param['t2v_dim'])(inp)
        x = LSTM(self.param['unit'], activation=self.param['act'])(x)
        x = Dense(1)(x)
        m = Model(inp, x)
        m.compile(loss='mse', optimizer='adam')
        m.summary()
        self.model = m
    
    def train(self, X_train, y_train):

        self.history = self.model.fit(X_train, y_train, epochs=20, validation_split=0.2, shuffle=False)
    
    def plot_training(self):

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def evaluate(self, X_test, y_test):

        self.model.evaluate(X_test, y_test)

    def plot_predictions(self, y_test, y_hat, item="AMAZON stock price"):

        plt.plot(y_test, label=f"Actual {item}", color='green')
        plt.plot(y_hat, label=f"Predicted {item}", color='red')
        
        plt.title(f'{item} prediction')
        plt.xlabel('Time [days]')
        plt.ylabel(f'{item}')
        plt.legend(loc='best')
        plt.show()

    

    

    
    

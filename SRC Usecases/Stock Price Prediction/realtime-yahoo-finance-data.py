import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

def gen_sequence(id_df, seq_length, seq_cols):
    
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

def gen_labels(id_df, seq_length, label):
    
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    
    return data_matrix[seq_length:num_elements, :]

class RealtimeFinanceData:

    def __init__(self, name='GOOGL', from_date='2016-01-01', to_date='2022-02-20') -> None:

        self.df = pdr.get_data_yahoo(name, from_date, to_date).dropna().reset_index()
    
    def generate_data(self, SEQ_LEN = 20, train_split=True):
        # pattern X is the size of Seq_len (e.g. use the first 20 days to predict 21st day)
        X, y = [], []
        for sequence in gen_sequence(self.df, SEQ_LEN, ['Open', 'Close']):
            X.append(sequence)
            
        for sequence in gen_labels(self.df, SEQ_LEN, ['Close']):
            y.append(sequence)
            
        X = np.asarray(X)
        y = np.asarray(y)

        if train_split == True:
            train_dim = int(0.7*len(self.df))
            X_train, X_test = X[:train_dim], X[train_dim:]
            y_train, y_test = y[:train_dim], y[train_dim:]

            return X_train, X_test, y_train, y_test

        else:
            return X, y

    def plot_close_price(self):
        ax = self.df.plot(x='Date' , y='Close');
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (USD)")
        plt.show()

if __name__ == "__main__":

    data_instance = RealtimeFinanceData()
    #print(data_instance.df)
    #data_instance.plot_close_price()
    
        
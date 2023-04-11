import numpy as np
import pandas as pd
from data import RealtimeFinanceData
from time2vec import Time2Vec

def predict_price(stocks=["GOOGL", "AMZN", "MSFT"], seq_len = 50):
    for stock in stocks:
        data_instance = RealtimeFinanceData(name=stock)
        #data_instance.plot_close_price()
        X_train, X_test, y_train, y_test = data_instance.generate_data(SEQ_LEN=seq_len)
        model_instance = Time2Vec(dim=seq_len)
        model_instance.set_LSTM()
        model_instance.train(X_train, y_train)
        y_hat = model_instance.model.predict(X_test)
        model_instance.plot_predictions(y_test, y_hat, item=f"{stock} stock price")


if __name__ == "__main__":
    predict_price()
"""

Pre-processing of the poloniex (crypto-currencies) data used to produce the input tensors of the neural network. <br>
For each stock, the input is a raw time series of the prices (High, Low, Open, Close). <i>Please note for crypto-currencies, the market never closes, so Close(t) = Open(t+1). </i><br>
The output is a matrix of 3 rows and n (number of available data points) columns. <br>
The columns correspond to:
- High(t-1)/Open(t-1)
- Low(t-1)/Open(t-1)
- Open(t)/Open(t-1)

We don't need to normalize the data since it's already of ratio of 2 prices closed to one.

"""

import os
import pandas as pd
import numpy as np

from tqdm import tqdm


def main():
    data_dir = "/poloniex_data/"
    directory = os.getcwd() + data_dir  # path to the files
    files_tags = os.listdir(directory)  # these are the differents pdf files

    # this is here because hidden files are also shown in the list.
    for file in files_tags:
        if file[0] == ".":
            files_tags.remove(file)
    stock_name = [file.split(".")[0] for file in files_tags]
    stocks = [file for file in files_tags]
    print(len(stock_name) == len(stocks))
    print("There are {} different currencies.".format(len(stock_name)))

    for s in stocks:
        df = pd.read_csv("." + data_dir + s)
        print(s, len(df))

    # We want roughly 1 year of data. So, we drop the data with less than 17000 rows.

    kept_stocks = [
        "ETCBTC.csv",
        "ETHBTC.csv",
        "DOGEBTC.csv",
        "ETHUSDT.csv",
        "BTCUSDT.csv",
        "XRPBTC.csv",
        "DASHBTC.csv",
        "XMRBTC.csv",
        "LTCBTC.csv",
        "ETCETH.csv",
    ]
    len_stocks = list()

    for s in kept_stocks:
        df = pd.read_csv("." + data_dir + s)
        len_stocks.append(len(df))

    min_len = np.min(len_stocks)

    list_open = list()
    list_close = list()
    list_high = list()
    list_low = list()

    for s in tqdm(kept_stocks):
        data = pd.read_csv(os.getcwd() + data_dir + s).fillna("bfill").copy()
        data = data[["open", "close", "high", "low"]]
        data = data.tail(min_len)
        list_open.append(data.open.values)
        list_close.append(data.close.values)
        list_high.append(data.high.values)
        list_low.append(data.low.values)

    array_open = np.transpose(np.array(list_open))[:-1]
    array_open_of_the_day = np.transpose(np.array(list_open))[1:]
    array_close = np.transpose(np.array(list_close))[:-1]
    array_high = np.transpose(np.array(list_high))[:-1]
    array_low = np.transpose(np.array(list_low))[:-1]

    np.transpose(np.array(list_low)).shape

    X = np.transpose(
        np.array(
            [
                array_high / array_open,
                array_low / array_open,
                array_open_of_the_day / array_open,
            ]
        ),
        axes=(0, 2, 1),
    )

    # The shape corresponds to:
    # - 3: Number of features
    # - 10: Number of stocks
    # - 17030: Number of data points

    np.save("./np_data/inputCrypto.npy", X)


if __name__ == "__main__":
    main()

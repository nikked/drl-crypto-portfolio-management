"""
Pre-processing of the S&P 500 data used to produce the input tensors of the neural network. <br>
For each stock, the input is a raw time series of the prices (High, Low, Open, Close).
The output is a matrix of 4 rows and n (number of available data points) columns. <br>
The columns correspond to:
- Close(t-1)/Open(t-1)
- High(t-1)/Open(t-1)
- Low(t-1)/Open(t-1)
- Open(t)/Open(t-1)

We don't need to normalize the data since it's already of ratio of 2 prices closed to one.

The shape corresponds to:
- 4: Number of features
- 5: Number of stocks we want to study
- 17030: Number of data points
"""

import os

from tqdm import tqdm

import numpy as np
import pandas as pd

PREFERRED_STOCK_INDECES = [3, 7, 12, 37, 42]
DATA_DIR = "/data/individual_stocks_5yr/"
OUT_PATH = "./data/np_data/input.npy"


def main(count_of_stocks=5, save=False):

    all_valid_stocks = _get_valid_stock_filepaths()

    kept_stock_rl = []

    for idx in range(count_of_stocks):
        stock_index = PREFERRED_STOCK_INDECES[idx]
        kept_stock_rl.append(all_valid_stocks[stock_index])

    stocks_tensor = _make_stocks_tensor(kept_stock_rl)

    if save:
        print("Saving stocks numpy tensor to path: {}".format(OUT_PATH))
        np.save(OUT_PATH, stocks_tensor)

    return stocks_tensor


def _get_valid_stock_filepaths():

    # This data dir contains daily stock returns of companies
    directory = os.path.join(os.getcwd() + DATA_DIR)  # path to the files
    stock_filepaths = os.listdir(directory)

    # this is here because hidden files are also shown in the list.
    for file in stock_filepaths:
        if file[0] == ".":
            stock_filepaths.remove(file)

    stock_filepaths = [file for file in stock_filepaths]

    print("In total there are {} different stocks.".format(len(stock_filepaths)))

    valid_stocks = list()
    not_valid_stocks = list()

    # Ignore stocks that do not have exactly 1259 trading days of history
    for stock in tqdm(stock_filepaths):
        if not stock.endswith("csv"):
            continue
        stock_df = pd.read_csv(os.getcwd() + DATA_DIR + stock)

        if len(stock_df) != 1259:

            not_valid_stocks.append(stock)
        else:
            valid_stocks.append(stock)

    print("Found {} valid stocks".format(len(valid_stocks)))

    return valid_stocks


def _make_stocks_tensor(kept_stock_rl):

    list_open = list()
    list_close = list()
    list_high = list()
    list_low = list()

    for kept_stock in tqdm(kept_stock_rl):
        data = pd.read_csv(os.getcwd() + DATA_DIR + kept_stock).fillna("bfill").copy()
        data = data[["open", "close", "high", "low"]]
        list_open.append(data.open.values)
        list_close.append(data.close.values)
        list_high.append(data.high.values)
        list_low.append(data.low.values)

    array_open = np.transpose(np.array(list_open))[:-1]
    array_open_of_the_day = np.transpose(np.array(list_open))[1:]
    array_close = np.transpose(np.array(list_close))[:-1]
    array_high = np.transpose(np.array(list_high))[:-1]
    array_low = np.transpose(np.array(list_low))[:-1]

    stocks_tensor = np.transpose(
        np.array(
            [
                array_close / array_open,
                array_high / array_open,
                array_low / array_open,
                array_open_of_the_day / array_open,
            ]
        ),
        axes=(0, 2, 1),
    )

    return stocks_tensor


if __name__ == "__main__":
    main()

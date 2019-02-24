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
"""

import os
from pprint import pprint

from tqdm import tqdm

import numpy as np
import pandas as pd


def main():  # pylint: disable=too-many-locals

    # This data dir contains dailly stock returns of companies
    data_dir = "/data/individual_stocks_5yr/"
    directory = os.getcwd() + data_dir  # path to the files
    files_tags = os.listdir(directory)

    # this is here because hidden files are also shown in the list.
    for file in files_tags:
        if file[0] == ".":
            files_tags.remove(file)

    stock_name = [file.split("_")[0] for file in files_tags]
    stocks = [file for file in files_tags]

    print("There are {} different stocks.".format(len(stock_name)))

    kept_stocks = list()
    not_kept_stocks = list()

    # Ignore stocks that do not have exactly 1259 trading days of history
    for stock in tqdm(stocks):
        if not stock.endswith("csv"):
            continue
        stock_df = pd.read_csv(os.getcwd() + data_dir + stock)

        if len(stock_df) != 1259:

            not_kept_stocks.append(stock)
        else:
            kept_stocks.append(stock)

    pprint(stock_df.head())

    print(not_kept_stocks)
    print(kept_stocks)

    kept_stock_rl = [
        kept_stocks[3],
        kept_stocks[7],
        kept_stocks[12],
        kept_stocks[37],
        kept_stocks[42],
    ]

    list_open = list()
    list_close = list()
    list_high = list()
    list_low = list()

    for kept_stock in tqdm(kept_stock_rl):
        data = pd.read_csv(os.getcwd() + data_dir + kept_stock).fillna("bfill").copy()
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

    # The shape corresponds to:
    # - 4: Number of features
    # - 5: Number of stocks we want to study
    # - 17030: Number of data points

    # # Save

    np.save("./data/np_data/input.npy", stocks_tensor)


if __name__ == "__main__":
    main()

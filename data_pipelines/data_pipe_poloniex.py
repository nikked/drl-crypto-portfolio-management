"""

Pre-processing of the poloniex (crypto-currencies) data used to produce the input tensors of the neural network. <br>
For each crypto, the input is a raw time series of the prices (High, Low, Open, Close). <i>Please note for crypto-currencies, the market never closes, so Close(t) = Open(t+1). </i><br>
The output is a matrix of 3 rows and n (number of available data points) columns. <br>
The columns correspond to:
- High(t-1)/Open(t-1)
- Low(t-1)/Open(t-1)
- Open(t)/Open(t-1)

We don't need to normalize the data since it's already of ratio of 2 prices closed to one.

    # The shape corresponds to:
    # - 3: Number of features
    # - 10: Number of cryptos
    # - 17030: Number of data points

"""

import os
from pprint import pprint

import pandas as pd
import numpy as np


def main(no_of_cryptos=5, max_count_of_periods=10000, save=False):
    data_dir = "/data/poloniex_data/"
    directory = os.getcwd() + data_dir  # path to the files
    files_tags = os.listdir(directory)  # these are the differents pdf files

    # this is here because hidden files are also shown in the list.
    for file in files_tags:
        if file[0] == "." or file == "f":
            files_tags.remove(file)

    crypto_names = [file.split(".")[0] for file in files_tags]
    cryptos = [file for file in files_tags]
    # print("There are {} different currencies.".format(len(crypto_names)))
    # pprint(crypto_names)

    # We want roughly 1 year of data. So, we drop the data with less than
    # 17000 rows.

    kept_cryptos = [
        "ETCBTC.csv",
        "ETHBTC.csv",
        "DOGEBTC.csv",
        # "ETHUSDT.csv",
        # "BTCUSDT.csv",
        "XRPBTC.csv",
        "DASHBTC.csv",
        "XMRBTC.csv",
        "LTCBTC.csv",
        # "ETCETH.csv",
    ][:no_of_cryptos]

    asset_list = ["ETC", "ETH", "DOGE", "XRP", "DASH", "XMR", "LTC"][:no_of_cryptos]

    len_cryptos = list()

    for idx, kept_crypto in enumerate(kept_cryptos):
        crypto_df = pd.read_csv("." + data_dir + kept_crypto)
        len_cryptos.append(len(crypto_df))

    min_len = np.min(len_cryptos)

    list_open = list()
    list_close = list()
    list_high = list()
    list_low = list()

    for crypto in kept_cryptos:

        if idx >= no_of_cryptos:
            break

        data = pd.read_csv(os.getcwd() + data_dir + crypto).fillna("bfill").copy()
        data = data[["open", "close", "high", "low"]]
        data = data.tail(min_len)
        list_open.append(data.open.values[:max_count_of_periods])
        list_close.append(data.close.values[:max_count_of_periods])
        list_high.append(data.high.values[:max_count_of_periods])
        list_low.append(data.low.values[:max_count_of_periods])

    array_open = np.transpose(np.array(list_open))[:-1]
    array_open_of_the_day = np.transpose(np.array(list_open))[1:]
    # array_close = np.transpose(np.array(list_close))[:-1]
    array_high = np.transpose(np.array(list_high))[:-1]
    array_low = np.transpose(np.array(list_low))[:-1]

    # np.transpose(np.array(list_low)).shape

    crypto_tensor = np.transpose(
        np.array(
            [
                array_high / array_open,
                array_low / array_open,
                array_open_of_the_day / array_open,
            ]
        ),
        axes=(0, 2, 1),
    )

    if save:
        np.save("./data/np_data/inputCrypto.npy", crypto_tensor)

    print("Returning dataset")
    print(asset_list)
    pprint(crypto_tensor.shape)
    print()

    return crypto_tensor, asset_list


if __name__ == "__main__":
    main()

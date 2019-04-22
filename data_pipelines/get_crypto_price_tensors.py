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

from data_pipelines.get_data_from_poloniex_api import download_crypto_data, DATA_DIR


def main(
    no_of_cryptos=5,
    start_date="20190101",
    end_date="20190319",
    trading_period_length="2h",
    train_session_name="none"
):

    cryptos_dict = {}

    # 11 workable
    # chosen_cryptos = ["XMR", "XRP", "LTC", "DASH", "DOGE", "NMC", "BTS", "NXT", "PPC", "MAID", "XCP", "ETH", "ETC"][:no_of_cryptos]

    if train_session_name == "long_run":
        chosen_cryptos = ["LTC", "XRP", "DASH", "DOGE", "NMC",
                          "BTS", "PPC", "MAID"][:no_of_cryptos]

    if train_session_name == "Jiang_et_al._backtest_period_3":
        # chosen_cryptos = ["ETH","PASC","XMR","reversed_USDT","LTC","XRP","ETC","MAID","FCT","DASH","ZEC"][:no_of_cryptos]
        chosen_cryptos = ["LTC", "ETH", "PASC", "XMR", "XRP",
                          "ETC", "MAID", "FCT", "DASH", "ZEC"][:no_of_cryptos]

    else:
        chosen_cryptos = ["XMR", "XRP", "LTC", "DASH", "ETH",
                          "MAID", "ETC",  "NMC", "BTS", "PPC", ][:no_of_cryptos]

    for crypto in chosen_cryptos:
        cryptos_dict[crypto] = os.path.join(
            f"BTC_{crypto}",
            f"{start_date}-{end_date}",
            f"BTC_{crypto}_{start_date}-{end_date}_{trading_period_length}.csv",
        )

    for crypto_ticker, crypto_data_fp in cryptos_dict.items():
        if not os.path.isfile(crypto_data_fp):
            download_crypto_data(
                f"BTC_{crypto_ticker}", start_date, end_date, trading_period_length
            )

    # Download bitcoin price for the period
    btc_price_fp = f"USDT_BTC_{start_date}-{end_date}_{trading_period_length}.csv"
    if not os.path.isfile(btc_price_fp):
        download_crypto_data(f"USDT_BTC", start_date, end_date, trading_period_length)

    chosen_crypto_fps = []

    for crypto in chosen_cryptos:
        chosen_crypto_fps.append(cryptos_dict[crypto])

    crypto_tensor = _make_crypto_tensor(chosen_crypto_fps, no_of_cryptos)

    print("Returning dataset")
    print(chosen_cryptos)
    pprint(crypto_tensor.shape)
    print()

    return crypto_tensor, chosen_cryptos


def _make_crypto_tensor(kept_cryptos, no_of_cryptos):
    
    list_open = list()
    list_close = list()
    list_high = list()
    list_low = list()

    # hacky, put a crypto with full history first and calculate its length
    # to be able to zero pad data
    first_crypto_list_len = 0

    for idx, crypto in enumerate(kept_cryptos):

        if idx >= no_of_cryptos:
            break

        data_fp = os.path.join(os.getcwd(), DATA_DIR, crypto)

        data = pd.read_csv(data_fp).fillna("bfill").copy()
        data = data[["open", "close", "high", "low"]]

        if not first_crypto_list_len:
            first_crypto_list_len = len(data.open.values)

        missing_datapoints = 0
        if len(data.open.values) < first_crypto_list_len:
            missing_datapoints = first_crypto_list_len - len(data.open.values)

        list_open.append(
            np.pad(data.open.values, (missing_datapoints, 0), 'edge'))
        list_close.append(
            np.pad(data.close.values, (missing_datapoints, 0), 'edge'))
        list_high.append(
            np.pad(data.high.values, (missing_datapoints, 0), 'edge'))
        list_low.append(
            np.pad(data.low.values, (missing_datapoints, 0), 'edge'))

    array_open = np.transpose(np.array(list_open))[:-1]
    array_open_of_the_day = np.transpose(np.array(list_open))[1:]
    array_high = np.transpose(np.array(list_high))[:-1]
    array_low = np.transpose(np.array(list_low))[:-1]

    for l in list_open:
        print(len(l))
        print(l[:100])

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

    return crypto_tensor


if __name__ == "__main__":
    main()

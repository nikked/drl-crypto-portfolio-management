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

    # Final backtest lists
    if train_session_name.startswith("Calm_before_the_storm"):
        print(f"Using assets for session {train_session_name}")
        chosen_cryptos = ['XMR', 'ETH', 'USDT', 'DASH', 'LTC', 'ETC', 'FCT', 'MAID', 'LSK', 'BTS', 'STEEM'][:no_of_cryptos]
    elif train_session_name.startswith("Awakening"):
        print(f"Using assets for session {train_session_name}")
        chosen_cryptos = ['XMR', 'ETH', 'USDT', 'DASH', 'XRP', 'ETC', 'ZEC', 'FCT', 'REP', 'STEEM', 'MAID'][:no_of_cryptos]
    elif train_session_name.startswith("Ripple_bullrun"):
        print(f"Using assets for session {train_session_name}")
        chosen_cryptos = ['XMR', 'ETH', 'USDT', 'DASH', 'XRP', 'LTC', 'ETC', 'MAID', 'FCT', 'GNT', 'ZEC'][:no_of_cryptos]
    elif train_session_name.startswith("Ethereum_valley"):
        print(f"Using assets for session {train_session_name}")
        chosen_cryptos = ['XMR', 'ETH', 'USDT', 'DASH', 'XRP', 'LTC', 'ETC', 'STR', 'XEM', 'DGB', 'ZEC'][:no_of_cryptos]
    elif train_session_name.startswith("All-time_high"):
        print(f"Using assets for session {train_session_name}")
        chosen_cryptos = ['XMR', 'ETH', 'USDT', 'DASH', 'XRP', 'LTC', 'ETC', 'BCH', 'STR', 'VTC', 'LSK'][:no_of_cryptos]
    elif train_session_name.startswith("Rock_bottom"):
        print(f"Using assets for session {train_session_name}")
        chosen_cryptos = ['XMR', 'ETH', 'USDT', 'DASH', 'XRP', 'LTC', 'BCH', 'STR', 'BCHSV', 'ZRX', 'ZEC'][:no_of_cryptos]
    elif train_session_name.startswith("Recent"):
        print(f"Using assets for session {train_session_name}")
        chosen_cryptos = ['XMR', 'ETH', 'USDT', 'DASH', 'XRP', 'LTC', 'STR', 'BCHABC', 'BCHSV', 'EOS', 'DGB'][:no_of_cryptos]

    else:
        print("\nWARNING USING DEFAULT ASSETS. Please ensure this is a test session")
        chosen_cryptos = ["XMR", "USDT", "XRP", "LTC", "DASH", "ETH",
                          "MAID", "ETC",  "NMC", "BTS", "PPC", ][:no_of_cryptos]

    for crypto in chosen_cryptos:
        if crypto == "USDT":
            continue
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
    btc_price_fn = f"USDT_BTC_{start_date}-{end_date}_{trading_period_length}.csv"
    if not os.path.isfile(btc_price_fn):
        download_crypto_data(f"USDT_BTC", start_date, end_date, trading_period_length)

    chosen_crypto_fps = []

    for crypto in chosen_cryptos:
        if crypto == 'USDT':

            btc_price_fp = os.path.join("USDT_BTC", f"{start_date}-{end_date}", btc_price_fn)
            chosen_crypto_fps.append(btc_price_fp)
        else:
            chosen_crypto_fps.append(cryptos_dict[crypto])

    crypto_tensor = _make_crypto_tensor(chosen_crypto_fps, no_of_cryptos)

    print("Returning dataset")
    pprint(crypto_tensor.shape)
    print()

    return crypto_tensor, chosen_cryptos


def _make_crypto_tensor(chosen_crypto_fps, no_of_cryptos):
    list_open = list()
    list_close = list()
    list_high = list()
    list_low = list()

    # hacky, put a crypto with full history first and calculate its length
    # to be able to zero pad data
    first_crypto_list_len = 0

    for idx, crypto_fp in enumerate(chosen_crypto_fps):

        if idx >= no_of_cryptos:
            break

        data_fp = os.path.join(os.getcwd(), DATA_DIR, crypto_fp)

        data = pd.read_csv(data_fp).fillna("bfill").copy()
        data = data[["open", "close", "high", "low"]]

        if not first_crypto_list_len:
            first_crypto_list_len = len(data.open.values)

        missing_datapoints = 0
        if len(data.open.values) < first_crypto_list_len:
            missing_datapoints = first_crypto_list_len - len(data.open.values)

        open_price = data.open.values
        close_price = data.close.values
        high_price = data.high.values
        low_price = data.low.values

        # invert USDT values
        if "USDT" in crypto_fp:
            open_price = np.true_divide(1, data.open.values)
            close_price = np.true_divide(1, data.close.values)
            high_price = np.true_divide(1, data.high.values)
            low_price = np.true_divide(1, data.low.values)

        list_open.append(
            np.pad(open_price, (missing_datapoints, 0), 'edge'))
        list_close.append(
            np.pad(close_price, (missing_datapoints, 0), 'edge'))
        list_high.append(
            np.pad(high_price, (missing_datapoints, 0), 'edge'))
        list_low.append(
            np.pad(low_price, (missing_datapoints, 0), 'edge'))

    array_open = np.transpose(np.array(list_open))[:-1]
    array_open_of_the_day = np.transpose(np.array(list_open))[1:]
    array_high = np.transpose(np.array(list_high))[:-1]
    array_low = np.transpose(np.array(list_low))[:-1]

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

import argparse
import numpy as np


from src.train_rl_algorithm import train_rl_algorithm
from src.test_rl_algorithm import test_rl_algorithm
from src.analysis import analysis
from src.environment import TradeEnv

from src.params import (
    LENGTH_TENSOR,
    PF_INITIAL_VALUE,
    TRADING_COST,
    INTEREST_RATE,
    RATIO_TRAIN,
    RATIO_VAL,
)


CRYPTO_DATA_FP = "./data/np_data/inputCrypto.npy"
STOCK_DATA_FP = "./data/np_data/input.npy"

DEFAULT_TRADE_ENV_ARGS = {
    "path": None,
    "window_length": LENGTH_TENSOR,
    "portfolio_value": PF_INITIAL_VALUE,
    "trading_cost": TRADING_COST,
    "interest_rate": INTEREST_RATE,
    "train_size": RATIO_TRAIN,
}


def main(
    interactive_session=False, crypto_data=False, gpu_device=None, verbose=True
):  # pylint: disable=too-many-locals

    if crypto_data:
        data_source_fp = CRYPTO_DATA_FP

    else:
        data_source_fp = STOCK_DATA_FP

    trade_env_args = DEFAULT_TRADE_ENV_ARGS
    trade_env_args["path"] = data_source_fp

    data_source = np.load(data_source_fp)
    nb_stocks = data_source.shape[1]

    nb_feature_map = data_source.shape[0]
    trading_period = data_source.shape[2]

    # HP of the problem
    # Total number of steps for pre-training in the training set
    total_steps_train = int(RATIO_TRAIN * trading_period)

    # Total number of steps for pre-training in the validation set
    total_steps_val = int(RATIO_VAL * trading_period)

    # Total number of steps for the test
    total_steps_test = trading_period - total_steps_train - total_steps_val

    data_type = data_source_fp.split("/")[2][5:].split(".")[0]

    asset_list = _get_asset_list(data_type, nb_stocks)

    # other environment Parameters

    w_eq = np.array(np.array([1 / (nb_stocks + 1)] * (nb_stocks + 1)))
    w_s = np.array(np.array([1] + [0.0] * nb_stocks))

    # Creation of the trading environment
    env, env_eq, env_s, action_fu, env_fu = _get_train_environments(
        nb_stocks, trade_env_args
    )

    # Agent training
    actor, state_fu, done_fu, list_final_pf, list_final_pf_eq, list_final_pf_s = train_rl_algorithm(
        interactive_session,
        env,
        env_eq,
        env_s,
        action_fu,
        env_fu,
        trade_env_args,
        w_eq,
        w_s,
        asset_list,
        total_steps_train,
        total_steps_val,
        nb_feature_map,
        nb_stocks,
        gpu_device,
        verbose,
    )

    # Agent evaluation
    p_list, p_list_eq, p_list_fu, p_list_s, w_list = test_rl_algorithm(
        actor,
        state_fu,
        done_fu,
        env,
        env_eq,
        env_s,
        action_fu,
        env_fu,
        total_steps_train,
        total_steps_val,
        total_steps_test,
        w_eq,
        w_s,
        nb_stocks,
    )

    # Analysis
    analysis(
        p_list,
        p_list_eq,
        p_list_s,
        p_list_fu,
        w_list,
        list_final_pf,
        list_final_pf_eq,
        list_final_pf_s,
        data_source_fp,
        total_steps_train,
        total_steps_val,
        nb_stocks,
    )


def _get_asset_list(data_type, nb_stocks):

    names_bio = ["JNJ", "PFE", "AMGN", "MDT", "CELG", "LLY"]
    names_utilities = ["XOM", "CVX", "MRK", "SLB", "MMM"]
    names_tech = [
        "FB",
        "AMZN",
        "MSFT",
        "AAPL",
        "T",
        "VZ",
        "CMCSA",
        "IBM",
        "CRM",
        "INTC",
    ]
    names_crypto = [
        "ETCBTC",
        "ETHBTC",
        "DOGEBTC",
        "ETHUSDT",
        "BTCUSDT",
        "XRPBTC",
        "DASHBTC",
        "XMRBTC",
        "LTCBTC",
        "ETCETH",
    ]

    # fix parameters of the network
    if data_type == "Utilities":
        asset_list = names_utilities
    elif data_type == "Bio":
        asset_list = names_bio
    elif data_type == "Tech":
        asset_list = names_tech
    elif data_type == "Crypto":
        asset_list = names_crypto
    else:
        asset_list = [i for i in range(nb_stocks)]

    return asset_list


def _get_train_environments(nb_stocks, trade_env_args):

    # environment for trading of the agent
    # this is the agent trading environment (policy network agent)

    env = TradeEnv(**trade_env_args)

    # environment for equiweighted
    # this environment is set up for an agent who only plays an equiweithed
    # portfolio (baseline)
    env_eq = TradeEnv(**trade_env_args)

    # environment secured (only money)
    # this environment is set up for an agentwho plays secure, keeps its money

    env_s = TradeEnv(**trade_env_args)

    # full on one stock environment
    # these environments are set up for agents who play only on one stock
    action_fu = list()
    env_fu = list()

    for i in range(nb_stocks):
        action = np.array([0] * (i + 1) + [1] + [0] * (nb_stocks - (i + 1)))
        action_fu.append(action)

        env_fu_i = TradeEnv(**trade_env_args)

        env_fu.append(env_fu_i)

    return env, env_eq, env_s, action_fu, env_fu


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        "-i",
        "--interactive_session",
        action="store_true",
        help="Plot interactively with matplotlib",
        default=False,
    )

    PARSER.add_argument(
        "-c",
        "--crypto_data",
        action="store_true",
        help="Use cryptocurrency data",
        default=False,
    )

    PARSER.add_argument(
        "-g", "--gpu_device", type=int, help="Choose GPU device number", default=None
    )
    PARSER.add_argument(
        "-v",
        "--verbose",
        help="Print train vectors",
        default=False,
        action="store_true",
    )

    ARGS = PARSER.parse_args()

    if ARGS.verbose:
        print("\nVerbose session. Alot of vectors will be printed below.\n")

    main(
        interactive_session=ARGS.interactive_session,
        crypto_data=ARGS.crypto_data,
        gpu_device=ARGS.gpu_device,
        verbose=ARGS.verbose,
    )

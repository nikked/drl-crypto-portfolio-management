# coding: utf-8

# # Introduction

# This notebook presents the main part of the project. It is decomposed in the following parts:
# - Parameters setting
# - Creation of the trading environment
# - Set-up of the trading agent (actor)
# - Set-up of the portfolio vector memory (PVM)
# - Agent training
# - Agent Evaluation
# - Analysis

import numpy as np
import argparse


from src.train_rl_algorithm import train_rl_algorithm
from src.test_rl_algorithm import test_rl_algorithm
from src.analysis import analysis
from src.environment import TradeEnv

from src.params import (
    PATH_DATA,
    n,
    nb_stocks,
    pf_init_train,
    trading_cost,
    interest_rate,
    RATIO_TRAIN,
    RATIO_VAL,
    BATCH_SIZE,
)


"""
TODO:
- clean params.py
    - datastuff out of params. data_type, data, trading_period, nb_feature_map, nb_stocks
    - big important params like trading cost to own big param

"""

DEFAULT_TRADE_ENV_ARGS = {
    "path": PATH_DATA,
    "window_length": n,
    "portfolio_value": pf_init_train,
    "trading_cost": trading_cost,
    "interest_rate": interest_rate,
    "train_size": RATIO_TRAIN,
}


def main(interactive_session=False):

    DATA_SOURCE = np.load(PATH_DATA)
    nb_stocks = DATA_SOURCE.shape[1]

    nb_feature_map = DATA_SOURCE.shape[0]
    trading_period = DATA_SOURCE.shape[2]

    # HP of the problem
    # Total number of steps for pre-training in the training set
    total_steps_train = int(RATIO_TRAIN * trading_period)

    # Total number of steps for pre-training in the validation set
    total_steps_val = int(RATIO_VAL * trading_period)

    # Total number of steps for the test
    total_steps_test = trading_period - total_steps_train - total_steps_val

    data_type = PATH_DATA.split("/")[2][5:].split(".")[0]

    list_stock = _get_list_stock(data_type)

    # other environment Parameters

    w_eq = np.array(np.array([1 / (nb_stocks + 1)] * (nb_stocks + 1)))
    w_s = np.array(np.array([1] + [0.0] * nb_stocks))

    # Creation of the trading environment
    env, env_eq, env_s, action_fu, env_fu = _get_train_environments()

    # Agent training
    actor, state_fu, done_fu, list_final_pf, list_final_pf_eq, list_final_pf_s = train_rl_algorithm(
        interactive_session,
        env,
        env_eq,
        env_s,
        action_fu,
        env_fu,
        DEFAULT_TRADE_ENV_ARGS,
        w_eq,
        w_s,
        list_stock,
        total_steps_train,
        total_steps_val,
        nb_feature_map,
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
        PATH_DATA,
        total_steps_train,
        total_steps_val,
    )


def _get_list_stock(data_type):

    # fix parameters of the network
    if data_type == "Utilities":
        list_stock = namesUtilities
    elif data_type == "Bio":
        list_stock = namesBio
    elif data_type == "Tech":
        list_stock = namesTech
    elif data_type == "Crypto":
        list_stock = namesCrypto
    else:
        list_stock = [i for i in range(nb_stocks)]

    return list_stock


def _get_train_environments():

    # environment for trading of the agent
    # this is the agent trading environment (policy network agent)

    env = TradeEnv(**DEFAULT_TRADE_ENV_ARGS)

    # environment for equiweighted
    # this environment is set up for an agent who only plays an equiweithed
    # portfolio (baseline)
    env_eq = TradeEnv(**DEFAULT_TRADE_ENV_ARGS)

    # environment secured (only money)
    # this environment is set up for an agentwho plays secure, keeps its money

    env_s = TradeEnv(**DEFAULT_TRADE_ENV_ARGS)

    # full on one stock environment
    # these environments are set up for agents who play only on one stock
    action_fu = list()
    env_fu = list()

    for i in range(nb_stocks):
        action = np.array([0] * (i + 1) + [1] + [0] * (nb_stocks - (i + 1)))
        action_fu.append(action)

        env_fu_i = TradeEnv(**DEFAULT_TRADE_ENV_ARGS)

        env_fu.append(env_fu_i)

    return env, env_eq, env_s, action_fu, env_fu


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        "-i",
        "--interactive_session",
        action="store_true",
        help="plot stuff and other interactive shit",
    )

    ARGS = PARSER.parse_args()

    if ARGS.interactive_session:
        main(interactive_session=True)
    else:
        main(interactive_session=False)


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
from environment import TradeEnv

from src.params import PATH_DATA, DEFAULT_TRADE_ENV_ARGS, m


def main(interactive_session=False):

    input_data_path = PATH_DATA

    env, env_eq, env_s, action_fu, env_fu = _get_train_environments()

    # Agent training
    actor, state_fu, done_fu, list_final_pf, list_final_pf_eq, list_final_pf_s = train_rl_algorithm(
        interactive_session, env, env_eq, env_s, action_fu, env_fu)

    # Agent evaluation
    p_list, p_list_eq, p_list_fu, p_list_s, w_list = test_rl_algorithm(
        actor, state_fu, done_fu, env, env_eq, env_s, action_fu, env_fu)

    # Analysis
    analysis(p_list, p_list_eq, p_list_s, p_list_fu, w_list,
             list_final_pf, list_final_pf_eq, list_final_pf_s,
             input_data_path)


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

    for i in range(m):
        action = np.array([0] * (i + 1) + [1] + [0] * (m - (i + 1)))
        action_fu.append(action)

        env_fu_i = TradeEnv(**DEFAULT_TRADE_ENV_ARGS)

        env_fu.append(env_fu_i)

    return env, env_eq, env_s, action_fu, env_fu


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        '-i',
        '--interactive_session',
        action="store_true",
        help="plot stuff and other interactive shit"
    )

    ARGS = PARSER.parse_args()

    if ARGS.interactive_session:
        main(interactive_session=True)
    else:
        main(interactive_session=False)

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


import argparse


from src.train_rl_algorithm import train_rl_algorithm
from src.test_rl_algorithm import test_rl_algorithm
from src.analysis import analysis


def main(interactive_session=False):

    # Agent training
    actor, state_fu, done_fu, list_final_pf, list_final_pf_eq, list_final_pf_s = train_rl_algorithm(
        interactive_session)

    # Agent evaluation
    p_list, p_list_eq, p_list_fu, p_list_s, w_list = test_rl_algorithm(
        actor, state_fu, done_fu)

    # Analysis
    analysis(p_list, p_list_eq, p_list_s, p_list_fu, w_list,
             list_final_pf, list_final_pf_eq, list_final_pf_s)


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

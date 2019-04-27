#!/bin/bash

while true;
    do
        python deep_rl_portfolio.py --calm_before_the_storm --trading_period_length 2h
        python deep_rl_portfolio.py --awakening --trading_period_length 2h
        python deep_rl_portfolio.py --ripple_bull_run --trading_period_length 2h
        python deep_rl_portfolio.py --ethereum_valley --trading_period_length 2h
        python deep_rl_portfolio.py --all_time_high --trading_period_length 2h
        python deep_rl_portfolio.py --rock_bottom --trading_period_length 2h
        python deep_rl_portfolio.py --recent --trading_period_length 2h
    done


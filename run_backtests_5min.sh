#!/bin/bash

while true; 
    do 
        python deep_rl_portfolio.py --calm_before_the_storm --trading_period_length 5min
        python deep_rl_portfolio.py --awakening --trading_period_length 5min
        python deep_rl_portfolio.py --ripple_bullrun --trading_period_length 5min
        python deep_rl_portfolio.py --ethereum_valley --trading_period_length 5min
        python deep_rl_portfolio.py --all-time_high --trading_period_length 5min
        python deep_rl_portfolio.py --rock_bottom --trading_period_length 5min
        python deep_rl_portfolio.py --recent --trading_period_length 5min
    done


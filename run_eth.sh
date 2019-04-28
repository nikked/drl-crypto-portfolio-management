#!/bin/bash

while true;
    do
        python deep_rl_portfolio.py --ethereum_valley --trading_period_length 15min
        python deep_rl_portfolio.py --ethereum_valley --trading_period_length 5min
        python deep_rl_portfolio.py --ethereum_valley --trading_period_length 30min
    done


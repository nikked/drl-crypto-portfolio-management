#!/bin/bash

while true;
    do
        python deep_rl_portfolio.py --awakening --trading_period_length 30min
        python deep_rl_portfolio.py --awakening --trading_period_length 15min
        python deep_rl_portfolio.py --awakening --trading_period_length 5min
    done


#!/bin/bash

while true; 
    do 
        python deep_rl_portfolio.py --calm_before_the_storm --trading_period_length 15min
        # python deep_rl_portfolio.py --awakening
        # python deep_rl_portfolio.py --ripple_bullrun
        # python deep_rl_portfolio.py --ethereum_valley
        # python deep_rl_portfolio.py --all-time_high
        # python deep_rl_portfolio.py --rock_bottom
        # python deep_rl_portfolio.py --recent
    done


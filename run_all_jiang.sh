#!/bin/bash

while true; 
    do 
        python deep_rl_portfolio.py -jbt1
        python deep_rl_portfolio.py -jbt2
        python deep_rl_portfolio.py -jbt3
        python deep_rl_portfolio.py -bull
        python deep_rl_portfolio.py -bear
        python deep_rl_portfolio.py -recent
        python deep_rl_portfolio.py -long
        python deep_rl_portfolio.py -nano
        python deep_rl_portfolio.py -pico
    done


# High Frequency Portfolio Optimization With Deep Reinforcement Learning
> This repository contains the code I built for my Master's thesis (Aalto University, Department of Finance). The thesis itself can be downloaded from [this link](https://github.com/nikked/rl_dl_gradu/raw/master/Linnansalo_Semi-High_Frequency_Portfolio_Optimization_With_Deep_Reinforcement_Learning.pdf).

My master's thesis explored deep reinforcement learning in algorithmic trading. I implemented a trading computer program that balances a portfolio of cryptocurrencies. The program tries to outperform an equally weighted strategy. More specifically, the program uses a convolutional neural network (CNN) built with Tensorflow.

## Table of Contents
1. [Usage and installation](#usage)
2. [Repository contents](#repo_contents)
3. [Thesis overview](#overview)

<a name="usage"/>

## Usage and installation
This repo requires Python 3.6+. For installing the dependencies, please run `pip install -r requirements.txt`


This repo exposes a single CLI: `python deep_rl_portfolio.py`. You do not need to worry about downloading datasets since they are automatically fetched from [Poloniex API](https://docs.poloniex.com/#introduction). (Provided that the Poloniex API still works as it did on April 2018).

I used this CLI to run the actual backtests of my thesis across different machines. Each backtest of my thesis has its' own keyword argument.

If you want to run one of my backtests, just type for example:
```
python deep_rl_portfolio.py --recent
```

To see the full list of arguments, run `python deep_rl_portfolio.py -h`



<a name="repo_contents"/>

## Repository contents

* `src` dir holds the models, environments etc. for the actual training
* `data` dir holds the datasets used (`.csv` files etc.)
* `data_pipelines` contains the scripts used to interact with external data APIs
* `tests` contains the unittests for this project


<a name="overview"/>

## Thesis overview


#### Algorithmic trading agent
At the core of the program is the algorithmic trading agent – a computer program powered by deep reinforcement learning. The agent follows some pre-determined instructions and executes market orders. Traditionally a human trader determines these instructions by using some technical indicators. I instead gave the trading agent raw price data as input and let it figure out its instructions.

The algorithmic trading agent has two goals. First, it chooses initial weights, and then it rebalances these weights periodically. Choosing proper initial weights is crucial since transaction costs make trade action costly. I evaluated the trading agent's performance in these two tasks by using two distinct agents: a static and a dynamic agent. The static agent only does the weight initialization and does not rebalance. The dynamic agent also rebalances. I found that the agent does a poor job in choosing initial weights.

In reinforcement learning terminology, the goal of the agent is to maximize the cumulative reward based on market actions. The cumulative reward is the final value of the portfolio at the end of the test period. The actions are portfolio weights. The trading agent chooses the next set of weights with a convolutional neural network (CNN) policy. The neural network is implemented with Tensorflow.

#### Cryptocurrencies as asset class
I chose cryptocurrencies as my underlying asset class. They are interesting to analyze due to high volatility and lack of previous research. The availability of data is also exceptional. Nevertheless, these same techniques could be utilized in other asset classes as well. I used the [Poloniex API](https://docs.poloniex.com/#introduction) as my data source.

#### Performance evaluation
I evaluated the performance of the agent in seven different backtest stories. Each backtest story reflects some unique and remarkable period in cryptocurrency history. One backtest period was from December 2017 when Bitcoin reached its all-time high price. Another one is from April 2017 when Bitcoin almost lost its place as the most valued cryptocurrency. The stories show the market conditions where the agent excels and reveals its risks.

The following figure visualizes my backtest periods against Bitcoin's dominance.

![Backtest periods](https://github.com/nikked/drl-crypto-portfolio-management/blob/master/images/backtest_choices.png)


#### Results
I found that the algorithmic trading agent closely follows an equally weighted strategy. This finding suggests that the agent is unavailable to decipher meaningful signals from the noisy price data. The machine learning approach does not provide an advantage over an equally weighted strategy. Nevertheless, the trading agent excels in volatile and mean-reverting market conditions. In these periods, the dynamic agent has lower volatility and a higher Sharpe ratio. However, it has a dangerous tendency to overinvest in a plummeting asset.

The following figure shows that the agent has a tendency to revert to equal weights (which is ~26% in this example).

![Mean reversion](https://github.com/nikked/drl-crypto-portfolio-management/blob/master/images/mean_reversion.png)

I also wanted to find out the optimal time-period for rebalancing for the dynamic agent. Therefore, I compared rebalancing periods from 15 minutes to 1 day. To make our results robust, I ran over a thousand simulations. I found that 15 – 30 minute rebalancing periods tend to work the best. This is visualized in the figure below where we can see that the 30 minute period had the highest Sharpe ratio.

![Rebalancing periods](https://github.com/nikked/drl-crypto-portfolio-management/blob/master/images/rebalancing_periods.png)


The results of the thesis contribute to the field of algorithmic finance. I showed that frequent rebalancing is a useful tool in the risk management of highly volatile asset classes. Further investigation is required to extend these findings beyond cryptocurrencies. For more details, please refer to [the complete work](https://github.com/nikked/rl_dl_gradu/raw/master/Linnansalo_Semi-High_Frequency_Portfolio_Optimization_With_Deep_Reinforcement_Learning.pdf).



### Credits
My thesis was heavily based on the paper "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem" by Jiang et. al. They have released their [source code on GitHub](https://github.com/ZhengyaoJiang/PGPortfolio). Also, I found [this IPython Notebook by selimamrouni very helpful](https://github.com/selimamrouni/Deep-Portfolio-Management-Reinforcement-Learning).
## High Frequency Portfolio Optimization With Deep Reinforcement Learning

> This repository contains the code I built for my Master's thesis (Aalto University, Department of Finance). The thesis itself can be downloaded from [this link](https://github.com/nikked/rl_dl_gradu/raw/master/Linnansalo_Semi-High_Frequency_Portfolio_Optimization_With_Deep_Reinforcement_Learning.pdf).

Given all the excitement around deep learning, I wanted to find out whether these machine learning methods could be applied to algorithmic portfolio management. More specifically, I implemented a reinforcement learning agent that periodically executes trades. I wanted to find out whether a trading program can outperform an equally weighted strategy.

I chose cryptocurrencies as my underlying asset class. This choice was mainly based on great data availability and my prior experience on trading cryptos. I used the [Poloniex API](https://docs.poloniex.com/#introduction) as my datasource.

The reinforcement learning framework was built with Tensorflow 1.9. The deep learning architecture used was based on the previous work of [Jiang et. al.](https://arxiv.org/abs/1706.10059)

For more details, please refer to [the completed work](https://github.com/nikked/rl_dl_gradu/raw/master/Linnansalo_Semi-High_Frequency_Portfolio_Optimization_With_Deep_Reinforcement_Learning.pdf).

## Usage
This repo exposes a single CLI: `deep_rl_portfolio.py`

To get started, please run a CLI with the `-h` flag. E.g.: `python deep_rl_portfolio.py -h` to get a list of acceptable flags.

## Installing
This repo requires Python 3.6+. For installing the dependencies, please run `pip install -r requirements.txt`


## Content

* `src` dir holds the models, environments etc. for the actual training
* `data` dir holds the datasets used (`.csv` files etc.)
* `data_pipelines` contains the scripts used to interact with external data APIs
* `tests` contains the unittests for this project


## Git precommit hooks
Make sure you are using the same git hooks as defined in .githooks!

Please run:
`git config core.hooksPath .githooks`


`chmod -R  744 .githooks`

This ensures that certain test procedures are ran before a commit is allowed


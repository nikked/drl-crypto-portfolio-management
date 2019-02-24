# Deep reinforcement learning in cryptocurrency price prediction

## Installing
This repo requires Python 3.6+. For installing the dependencies, please run `pip install -r requirements.txt`

## Usage
This repo exposes two CLIs:

* `fetch_data_from_external_source.py`: This CLI is used to download data to your local from various sources.
* `train_test_analyse_rl_algorithm.py`: This CLI controls the main workhorse of this repo. The actual training process and its validation.

To get started, please run a CLI with the `-h` flat. E.g.: `python fetch_data_from_external_source.py -h` to get a list of acceptable flags.

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


## Credits

Since this is my first work related to deep reinforcement learning, I was strongly inspired by work previously done by other authors. I would like to thank the following entities for making there code publicly available:

* [The original authors of article "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"](https://github.com/ZhengyaoJiang/PGPortfolio)
* [Mike Clark's further analysis based on the article above](https://github.com/wassname/rl-portfolio-management)
* [Selim Amrouni's further analysis based on the article above](https://github.com/selimamrouni/Deep-Portfolio-Management-Reinforcement-Learning)


## TODO
* feature: a small test dataset to run through full pipeline
    * can be run with unittests to rapidly check scenarios
* refactor train_test_analyse.py
* write docstrings to each function and refactor further
* write a small unit test that is run on a precommit hook
* improve requirements.txt : decide an specific env that the code is ran on
* add coverage to git hook

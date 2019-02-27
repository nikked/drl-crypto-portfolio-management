# Deep reinforcement learning in cryptocurrency price prediction

## Installing
This repo requires Python 3.6+. For installing the dependencies, please run `pip install -r requirements.txt`

## Usage
This repo exposes a single CLI:

* `deep_rl_portfolio.py`: This CLI is used to download data to your local from various sources.


To get started, please run a CLI with the `-h` flag. E.g.: `python deep_rl_portfolio.py -h` to get a list of acceptable flags.
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


## TODO
* write docstrings to each function and refactor further
* improve requirements.txt : decide an specific env that the code is ran on
* add tensorboard
* improve train plotting

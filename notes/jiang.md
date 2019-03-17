## Abstract

constnt redistribution of fund into different products

financial-model free, no assumptions or technical indicators needed

deep reinforcement learning

* Framework
    * EIIE ensemble of identical independent estimators
    * PVM: holds prev weights
    * Online stochastic batch learning scheme (?)
    * fully exploiting and explicit reward function (?)

* comparison to other portfolio selection strategies
* three back-test experiments, 30 min crypto

* claim that all three instances (CNN, RNN, LSTM) outperform, notoriously hard to replicate

## Intro
* general purpose cont deep RL framework, actor-critic FPG algo (silver 2014)
* Deng not applicable since limit ed to sinlge asset trading
* IEE: 
    * a NN whose job is to inspect the history of an asset and evaluate its potential for growth in the immediate future
    * evaluation score of each asset is discounted by the size of its intentional weight change and presented to a softmax layer whose outcome determines weights for next period
    * these weights are the market action for the agent
* PVM, holds portfolio weights
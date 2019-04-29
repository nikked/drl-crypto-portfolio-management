# Discussion:
* Random choice by DL network: We should use same asset allocation on each on each backtest
* this should be controlled 
* if another test setting, would control with same weights across periods. however wasnted to test DL strategy
* cannot replicate JIang

# Results




## Calm before the storm

> Dynamic sucked (except on 4h). Equal best (large weight on Bitcoin, low MDD)

5min example image

### Portfolio value
> Losses


* From the static agent we can observe that the DL agent made larger allocations on average to valuable assets on period 5min
* We can see that the 5 min - 2h trading action destroyed value. 
    * especially 5min trading was invaluable
* However 4h adjustments worked on average
* again 1d adjustments destroyed value 
* after tx costs

### MDDs
> moderate MDDs

* Higher frequencey, lower  mdd. (makes sense)
* mdd grows until 1d, this is probably because the extreme values where not near 00:00

### Shapre
> Lowest sharpes of all backtests

* best PF value on 1d, different asset allocation on average
* while 15 min has higher sharpe, it is due to less volatile average allocations
* 1d adjustments indeed destroyed value, while higher sharpe, static grew alot


### Jiang (30 min)

* 4-30x returns... how possible witht this price action
* once again, MDD at same ballpark
* how would they beat the best stock?


### Hack
* EQ
    * 5 min values, highest sharpe lowest mdd


## Awakening
> more volatility, dynamic slightly outperformed static on 15min + periods


2h or 4h example image 


### Portfolio value
> Moderate returns

* 5 min trade action destroyd value
* however, 15 min to 1d seems to create value on average
    * maybe 

### MDDs
> low mdds, prices stayed the same

* contrarily, mdd decreased when period increased
* might have to do with asset selection
* dynamic had lowest mdd, static highesst

### Sharpe
> Nice sharpes due to low volatility in crypto standards

* equals slight higher investment in BTC leaded to highes sharpe
* however higher MDD than dynamic, hard to say which was least risky

### Jiang (30 min)
* mdd much lowe, 0.17, they had 0.216
* same fapv as their benchmarks.
* I do not understand how they claimed to get 4-8x returns with this price acion

### Hack
* EQ
    * 2h 4h all fine ( 2h preferred)



## Ripple bullrun 

> Lot of mean reverting action, Dynamic outperformed all agents. THis is the optimal condition for agent

### Portfolio value
> Great returns

* DYnamic is able to cash profits (except on 5h, poor allocations)
* This shows why DL is a poor way of choosing weights... stochasticity
* all trading action create value, most in 15 - 30 min and 1d


### MDDs
> moderate MDDs

* Dynamic has strictly lowes MDD


### Sharpe
> Impressive sharpes

* each agent has great perf

### Jiang (30 min)
* MDD on the same ballpark 0.38 (very close)
* samish return as with icnn (old model)
* 3.2 vs 3.9
* fapv of 30 - 40 not achieved

### Hack
* EQ 30 min


## Ethereum valley

> Lot of mean reverting action, Dynamic outperformed all agents. THis is the optimal condition for agent

### Portfolio value
> Minimal returns

* trading strictstly creates vale. most on 15min -4h

### MDDs
> very high MDDs

* Ccontrarily to others, dynamic has highest MDD

### Sharpe
> low sharpes, deliberately chosen to be a valley

* Highest sharpe

### Hack
* EQ 15 min


## All-time high
* hard to say if trading destroys value
* ripple and eth stay on higher positions]

### Portfolio value
> Good returns since BTC goes to all time low


* dynamic looses since sells ripple and eht
* static is winner since lower cash weight

### MDDs
> very high MDDS

* similar to Ripple bullrun
* dynamic has strictly lowest MDDs

### Sharpe
> High sharpes

* Dynamic has best sharpe eventhough it sucks in returls

### Hack
* EQ: 30 min (2h or even 4h also fine)
* NOTE: replace static / dynamic 15 min if not corrected



## Rock bottom

> MEan reverting since trading action creatse value

### Portfolio value
* 5 min trading action destroys value
* 2h trading action 



### MDDs
> Low mdds


* Dynamic benefits from mean reverting and creates amore stable though very similar return

### Sharpe
> Great shrpes

* Low volatility of agent rewarded


### Hack
* EQ 4h



## Recent

> Trading action destroys value, no mean reversion

### Portfolio value
> Dynamic agent has lowest return

* Equal has highgest return as Bitcoin rocks
* Trading strictly destroyes value

### MDDs
> Low mdds

* dyn mdd lower than static as expected.
* eq has lowest due to high btc weight

### Sharpe
> Lowest sharpes due to BTC loss

* trading destroyed value


### Hack
* EQ 15 min

### Notes
* 5min had high btc weight


## Taken
* 5min (double)
* 15 min
* 30 min (double!!)
* 2h
* 4h

## Not taken
* 1d
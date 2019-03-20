## PGs

The key idea underlying policy gradients is to push up the probabilities of actions that lead to higher return, and push down the probabilities of actions that lead to lower return, until you arrive at the optimal policy.

 A PG algorithms optimize the parameters of a policy by following the gradients toward higher rewards. 


## four steps of pg
1. First, let the neural network policy play the game several times/ episodes and at each step compute the gradients that would make the chosen action even more likely, but don’t apply these gradients yet.

2. Once you have run several episodes, compute each action’s discounted score

3. If an action’s score is positive, it means that the action was good and you want to apply the gradients computed earlier to make the action even more likely to be chosen in the future. However, if the score is negative, it means the action was bad and you want to apply the opposite gradients to make this action slightly less likely in the future. The solution is simply to multiply each gradient vector by the corresponding action’s score.

4. Finally, compute the mean of all the resulting gradient vectors, and use it to perform a Gradient Descent step.


## RL: What two steps are required to form an expression of the policy gradient that can be numerically computed?
1) deriving the analytical gradient of policy performance, which turns out to have the form of an expected value, and then 

2) forming a sample estimate of that expected value, which can be computed with data from a finite number of agent-environment interaction steps.


## We like sigmoid because its derivative is non zero everywhere. Also relu. dying (elu)
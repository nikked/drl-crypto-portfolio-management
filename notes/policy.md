# Policy
* initialized at train_rl_algorithm as agent

## train_rl_algorithm
* action = agent.compute_w(x_current, w_previous)
*     agent.train(
        np.array(train_session_tracker["policy_x_t"]),
        np.array(train_session_tracker["policy_prev_weights"]),
        np.array(train_session_tracker["policy_prev_value"]),
        np.array(train_session_tracker["policy_daily_return_t"]),
    )

## test_rl_algorithm
* uses agents's compute_w function that computes actions:
        action = agent.compute_w(x_current, w_previous)
* 


window length affects no of periods trained
* take btc data only for test period
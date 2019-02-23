from src.params import nb_stocks, n, pf_init_test

import numpy as np


def test_rl_algorithm(
    actor,
    state_fu,
    done_fu,
    env,
    env_eq,
    env_s,
    action_fu,
    env_fu,
    total_steps_train,
    total_steps_val,
    total_steps_test,
    w_eq,
    w_s,
):
    #######TEST#######

    w_init_test = np.array(np.array([1] + [0] * nb_stocks))

    # initialization of the environment
    state, done = env.reset(w_init_test, pf_init_test, t=total_steps_train)

    state_eq, done_eq = env_eq.reset(w_eq, pf_init_test, t=total_steps_train)
    state_s, done_s = env_s.reset(w_s, pf_init_test, t=total_steps_train)

    for i in range(nb_stocks):
        state_fu[i], done_fu[i] = env_fu[i].reset(
            action_fu[i], pf_init_test, t=total_steps_train
        )

    # first element of the weight and portfolio value
    p_list = [pf_init_test]
    w_list = [w_init_test]

    p_list_eq = [pf_init_test]
    p_list_s = [pf_init_test]

    p_list_fu = list()
    for i in range(nb_stocks):
        p_list_fu.append([pf_init_test])

    pf_value_t_fu = [0] * nb_stocks

    for k in range(
        total_steps_train + total_steps_val - int(n / 2),
        total_steps_train + total_steps_val + total_steps_test - n,
    ):
        X_t = state[0].reshape([-1] + list(state[0].shape))
        W_previous = state[1].reshape([-1] + list(state[1].shape))
        pf_value_previous = state[2]
        # compute the action
        action = actor.compute_W(X_t, W_previous)
        # step forward environment
        state, reward, done = env.step(action)
        state_eq, reward_eq, done_eq = env_eq.step(w_eq)
        state_s, reward_s, done_s = env_s.step(w_s)

        for i in range(nb_stocks):
            state_fu[i], _, done_fu[i] = env_fu[i].step(action_fu[i])

        X_next = state[0]
        W_t = state[1]
        pf_value_t = state[2]

        pf_value_t_eq = state_eq[2]
        pf_value_t_s = state_s[2]
        for i in range(nb_stocks):
            pf_value_t_fu[i] = state_fu[i][2]

        dailyReturn_t = X_next[-1, :, -1]
        if k % 20 == 0:
            print("current portfolio value", round(pf_value_previous, 0))
            print("weights", W_previous)
        p_list.append(pf_value_t)
        w_list.append(W_t)

        p_list_eq.append(pf_value_t_eq)
        p_list_s.append(pf_value_t_s)
        for i in range(nb_stocks):
            p_list_fu[i].append(pf_value_t_fu[i])

        # here to breack the loop/not in original code
        if k == total_steps_train + total_steps_val - int(n / 2) + 100:
            break

    return p_list, p_list_eq, p_list_fu, p_list_s, w_list

import numpy as np

from src.params import PF_INIT_TEST, LENGTH_TENSOR


def test_rl_algorithm(  # pylint:  disable=too-many-arguments, too-many-locals
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
    no_of_assets,
):

    weights_equal = np.array(np.array([1 / (no_of_assets + 1)] * (no_of_assets + 1)))
    weights_single = np.array(np.array([1] + [0.0] * no_of_assets))

    w_init_test = np.array(np.array([1] + [0] * no_of_assets))

    # initialization of the environment
    state, _ = env.reset(w_init_test, PF_INIT_TEST, index=total_steps_train)

    state_eq, _ = env_eq.reset(weights_equal, PF_INIT_TEST, index=total_steps_train)
    state_s, _ = env_s.reset(weights_single, PF_INIT_TEST, index=total_steps_train)

    for i in range(no_of_assets):
        state_fu[i], done_fu[i] = env_fu[i].reset(
            action_fu[i], PF_INIT_TEST, index=total_steps_train
        )

    # first element of the weight and portfolio value
    p_list = [PF_INIT_TEST]
    w_list = [w_init_test]

    p_list_eq = [PF_INIT_TEST]
    p_list_s = [PF_INIT_TEST]

    p_list_fu = list()
    for i in range(no_of_assets):
        p_list_fu.append([PF_INIT_TEST])

    pf_value_t_fu = [0] * no_of_assets

    for k in range(
        total_steps_train + total_steps_val - int(LENGTH_TENSOR / 2),
        total_steps_train + total_steps_val + total_steps_test - LENGTH_TENSOR,
    ):
        x_current = state[0].reshape([-1] + list(state[0].shape))
        w_previous = state[1].reshape([-1] + list(state[1].shape))
        pf_value_previous = state[2]
        # compute the action
        action = actor.compute_w(x_current, w_previous)
        # step forward environment
        state, _, _ = env.step(action)
        state_eq, _, _ = env_eq.step(weights_equal)
        state_s, _, _ = env_s.step(weights_single)

        for i in range(no_of_assets):
            state_fu[i], _, done_fu[i] = env_fu[i].step(action_fu[i])

        # x_next = state[0]
        w_current = state[1]
        pf_value_t = state[2]

        pf_value_t_eq = state_eq[2]
        pf_value_t_s = state_s[2]
        for i in range(no_of_assets):
            pf_value_t_fu[i] = state_fu[i][2]

        # dailyReturn_t = x_next[-1, :, -1]
        if k % 20 == 0:
            print("current portfolio value", round(pf_value_previous, 0))
            print("weights", w_previous)
        p_list.append(pf_value_t)
        w_list.append(w_current)

        p_list_eq.append(pf_value_t_eq)
        p_list_s.append(pf_value_t_s)
        for i in range(no_of_assets):
            p_list_fu[i].append(pf_value_t_fu[i])

        # here to breack the loop/not in original code
        if k == total_steps_train + total_steps_val - int(LENGTH_TENSOR / 2) + 100:
            break

    return p_list, p_list_eq, p_list_fu, p_list_s, w_list

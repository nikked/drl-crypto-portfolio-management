import numpy as np
import matplotlib.pyplot as plt

from src.environment import TradeEnv
from src.params import (
    n,
    m,
    pf_init_train,
    trading_cost,
    interest_rate,
    dict_hp_pb,
    w_init_test,
    pf_init_test,
    total_steps_train,
    total_steps_val,
    list_stock,
    DEFAULT_TRADE_ENV_ARGS
)


def eval_perf(e, actor, render_plots):
    """
    This function evaluates the performance of the different types of agents.


    """
    list_weight_end_val = list()
    list_pf_end_training = list()
    list_pf_min_training = list()
    list_pf_max_training = list()
    list_pf_mean_training = list()
    list_pf_dd_training = list()

    #######TEST#######
    # environment for trading of the agent
    env_eval = TradeEnv(**DEFAULT_TRADE_ENV_ARGS)

    # initialization of the environment
    state_eval, done_eval = env_eval.reset(
        w_init_test, pf_init_test, t=total_steps_train
    )

    # first element of the weight and portfolio value
    p_list_eval = [pf_init_test]
    w_list_eval = [w_init_test]

    for k in range(total_steps_train, total_steps_train + total_steps_val - int(n / 2)):
        X_t = state_eval[0].reshape([-1] + list(state_eval[0].shape))
        W_previous = state_eval[1].reshape([-1] + list(state_eval[1].shape))
        pf_value_previous = state_eval[2]
        # compute the action
        action = actor.compute_W(X_t, W_previous)
        # step forward environment
        state_eval, reward_eval, done_eval = env_eval.step(action)

        X_next = state_eval[0]
        W_t_eval = state_eval[1]
        pf_value_t_eval = state_eval[2]

        dailyReturn_t = X_next[-1, :, -1]
        # print('current portfolio value', round(pf_value_previous,0))
        # print('weights', W_previous)
        p_list_eval.append(pf_value_t_eval)
        w_list_eval.append(W_t_eval)

    list_weight_end_val.append(w_list_eval[-1])
    list_pf_end_training.append(p_list_eval[-1])
    list_pf_min_training.append(np.min(p_list_eval))
    list_pf_max_training.append(np.max(p_list_eval))
    list_pf_mean_training.append(np.mean(p_list_eval))

    list_pf_dd_training.append(_get_max_draw_down(p_list_eval))

    print("End of test PF value:", round(p_list_eval[-1]))
    print("Min of test PF value:", round(np.min(p_list_eval)))
    print("Max of test PF value:", round(np.max(p_list_eval)))
    print("Mean of test PF value:", round(np.mean(p_list_eval)))
    print("Max Draw Down of test PF value:",
          round(_get_max_draw_down(p_list_eval)))
    print("End of test weights:", w_list_eval[-1])

    if render_plots:
        plt.title("Portfolio evolution (validation set) episode {}".format(e))
        plt.plot(p_list_eval, label="Agent Portfolio Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.show()
        plt.title("Portfolio weights (end of validation set) episode {}".format(e))
        plt.bar(np.arange(m + 1), list_weight_end_val[-1])
        plt.xticks(np.arange(m + 1), ["Money"] + list_stock, rotation=45)
        plt.show()

    names = ["Money"] + list_stock
    w_list_eval = np.array(w_list_eval)

    if render_plots:
        for j in range(m + 1):
            plt.plot(w_list_eval[:, j],
                     label="Weight Stock {}".format(names[j]))
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)
        plt.show()

def _get_max_draw_down(xs):
    xs = np.array(xs)
    i = np.argmax(np.maximum.accumulate(xs) - xs)  # end of the period
    j = np.argmax(xs[:i])  # start of period

    return xs[j] - xs[i]

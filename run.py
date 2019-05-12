import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto
from algorithms import Zooming, ADTM, ADMM


def simulate(algorithms, a, alpha, T, trials):
    cum_regret = np.zeros((len(algorithms), T + 1))
    for trial in range(trials):
        inst_regret = np.zeros((len(algorithms), T + 1))
        for alg in algorithms:
            alg.initialize()

        for t in range(1, T + 1):
            for i, alg in enumerate(algorithms):
                idx = alg.output()
                arm = alg.active_arms[idx]
                inst_regret[i, t] = min(abs(arm - 0.4), abs(arm - 0.8))
                y = a - min(abs(arm - 0.4), abs(arm - 0.8)) + pareto.rvs(alpha) - alpha / (alpha - 1)
                alg.observe(t, y)

        cum_regret += np.cumsum(inst_regret, axis=-1)
    return cum_regret / trials


def run_experiment(a):
    # configure parameters of experiments
    T = 20000
    trials = 40
    delta = 0.1
    alpha = 3.1
    epsilon = 1

    # compute upper bounds for moments of different orders
    a_hat = max(abs(a), abs(a - 0.4))
    sigma_second = max(alpha / ((alpha - 1) ** 2 * (alpha - 2)), 1 / (36 * np.sqrt(2)))
    nu_second = max(a_hat ** 2 + sigma_second, np.power(12 * np.sqrt(2), -(1 + epsilon)))
    nu_third = a_hat ** 3 + 2 * alpha * (alpha + 1) / (
            (alpha - 1) ** 3 * (alpha - 2) * (alpha - 3)) + 3 * a_hat * sigma_second

    # simulate
    c_zooming = 0.01  # searched within {1, 0.1, 0.01} and `0.01` is the best choice
    c_ADTM = 0.1  # searched within {1, 0.1, 0.01} and `0.1` is the best choice
    c_ADMM = 0.1  # searched within {1, 0.1, 0.01} and `0.1` is the best choice
    algorithms = [Zooming(delta, T, c_zooming, nu_third), ADTM(delta, T, c_ADTM, nu_second, epsilon),
                  ADMM(delta, T, c_ADMM, sigma_second, epsilon)]
    cum_regret = simulate(algorithms, a, alpha, T, trials)

    # plot figure
    plt.figure(figsize=(7, 4))
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    names = [f'{alg.__class__.__name__}' for alg in algorithms]
    linestyles = ['-', '--', '-.']
    for result, name, linestyle in zip(cum_regret, names, linestyles):
        plt.plot(result, label=name, linewidth=2.0, linestyle=linestyle)
    plt.legend(loc='upper left', frameon=True, fontsize=10)
    plt.xlabel('t', labelpad=1, fontsize=15)
    plt.ylabel('cumulative regret', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(f'cum_regret_{a}.png', dpi=500, bbox_inches='tight')


if __name__ == '__main__':
    run_experiment(a=0)
    run_experiment(a=-2)

# 最急降下法
import numpy as np
from hawkes_process import HawkesProcess


def grad_decent(objective: HawkesProcess, eta: float):
    """
    maximum likelihood estimation using gradient descent(steepest descent)
    """
    x = np.array([0.01, 0.5, 0.5])  # 初期値
    for i in range(100000):
        grad = objective.nll_grad(x)
        x = x - eta * grad
    return x


def main():
    T = 300
    t = np.array([
        2, 3, 5, 20, 23, 25, 40, 45, 46, 60, 61, 63, 99, 101, 105, 150, 155, 156, 160, 161, 162, 163, 164, 165, 166,
        190, 192, 193, 194, 210, 241, 242, 243, 245, 280, 282, 283, 284, 288
    ])

    hawkes_process = HawkesProcess(t, T)
    # maximum likelihood estimation
    theta = grad_decent(hawkes_process, 0.0001)

    print(theta)
    hawkes_process.plot_intensity(theta)
    hawkes_process.plot_intensity_accumulation(theta)


if __name__ == '__main__':
    main()

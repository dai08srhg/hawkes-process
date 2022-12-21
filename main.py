import numpy as np
from hawkes_process import HawkesProcess


def main():
    t = np.array([
        2, 3, 5, 20, 23, 25, 40, 45, 46, 60, 61, 63, 99, 101, 105, 150, 155, 156, 160, 161, 162, 163, 164, 165, 166,
        190, 192, 193, 194, 210, 241, 242, 243, 245, 280, 282, 283, 284, 288
    ])

    hawkes_process = HawkesProcess()
    # maximum likelihood estimation
    T = 300
    hawkes_process.fit(t, T)
    print(hawkes_process.theta)


if __name__ == '__main__':
    main()

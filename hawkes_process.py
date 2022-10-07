import numpy as np
import matplotlib.pyplot as plt
import itertools


class HawkesProcess:
    """
    Hawkes process
    intensity_func g(τ) = a*b*exp(-b*τ)
    """

    def __init__(self, t: np.array, T: int) -> None:
        """
        Args:
            t: Event occurrence time set
            T: Observation period length
        """
        self.t = t
        self.T = T
        self.n = len(t)

        self.dim = 3  # パラメータ数

    def nll(self, theta: np.array):
        """
        negative log-likelihood

        Args:
            theta: paramaters (μ, a, b)
        """
        mu = theta[0]
        a = theta[1]
        b = theta[2]

        G = {0: 0}
        for i in range(self.n - 1):
            G[i + 1] = (G[i] + a * b) * np.exp(-b * (self.t[i + 1] - self.t[i]))
        x = 0
        y = 0
        for i in range(self.n):
            x += np.log(mu + G[i])
            y += a * (1 - np.exp(-b * (self.T - self.t[i])))
        return -(x - (mu * self.T + y))

    def nll_grad(self, theta: np.array):
        """
        gradient of negative log-likelihood

        Args:
            theta: paramaters (μ, a, b)
        """
        mu = theta[0]
        a = theta[1]
        b = theta[2]

        # 一次変数
        G = {0: 0}
        G_db = {0: 0}
        lam = {0: mu}
        lam_da = {0: 0}
        lam_db = {0: 0}

        # 一次変数の計算
        # yapf: disable
        for i in range(self.n - 1):
            G[i + 1] = (G[i] + a * b) * np.exp(-b * (self.t[i + 1] - self.t[i]))
            G_db[i + 1] = (G_db[i] + a) * np.exp(-b * (self.t[i + 1] - self.t[i])) - G[i + 1] * (self.t[i + 1] - self.t[i])
            lam[i + 1] = G[i + 1] + mu
            lam_da[i + 1] = G[i + 1] / a
            lam_db[i + 1] = G_db[i + 1]
        # yapf: enable

        # 勾配の計算
        x = 0
        y = 0
        z = 0
        o = 0
        p = 0
        for i in range(self.n):
            x += (1 / lam[i])
            y += (1 / lam[i]) * lam_da[i]
            z += 1 - np.exp(-b * (self.T - self.t[i]))
            o += (1 / lam[i]) * lam_db[i]
            p += a * (self.T - self.t[i]) * np.exp(-b * (self.T - self.t[i]))

        grad_mu = x - self.T
        grad_a = y - z
        grad_b = o - p
        return np.array([-grad_mu, -grad_a, -grad_b])

    def plot_event(self):
        """
        plot event
        """
        fig, ax = plt.subplots(figsize=(12, 3))
        plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False, left=False)
        for t_i in self.t:
            ax.vlines(t_i, 0, 1)
        ax.set_ylim(0.0, 1.5)  # x軸の範囲
        ax.set_xlim(0.0, self.T)
        ax.set_xlabel('t')
        plt.show()

    def plot_intensity(self, theta):
        """
        plot intensity

        Args:
            theta: paramaters (μ, a, b)
        """
        mu = theta[0]
        a = theta[1]
        b = theta[2]

        strengths = []
        ts = []
        for t in range(self.T):
            h = self.t[self.t < t]
            strength = mu
            for t_j in h:
                strength += a * b * np.exp(-b * (t - t_j))
            strengths.append(strength)
            ts.append(t)

        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        ax1.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False, left=False)
        for t_i in self.t:
            ax1.vlines(t_i, 0, 1)
        ax1.set_title('event')
        ax1.set_ylim(0.0, 1.5)  # x軸の範囲
        ax1.set_xlim(0.0, self.T)
        ax1.set_xlabel('t')
        ax2.set_title('intensity')
        ax2.plot(ts, strengths)
        ax2.set_xlim(0.0, self.T)
        ax2.set_xlabel('t')
        plt.tight_layout()  # 追加
        plt.show()

    def plot_intensity_accumulation(self, theta):
        """
        plot intensity accumulation

        Args:
            theta: paramaters (μ, a, b)
        """
        mu = theta[0]
        a = theta[1]
        b = theta[2]
        totals = []
        total = 0
        strengths = []
        ts = []
        for t in range(self.T):
            h = self.t[self.t < t]
            strength = mu
            for t_j in h:
                strength += a * b * np.exp(-b * (t - t_j))
            strengths.append(strength)

            if t in self.t:
                total += 1
            ts.append(t)
            totals.append(total)

        accumulations = [i for i in itertools.accumulate(strengths)]
        fig = plt.figure(figsize=(5, 5))
        plt.title('Expected value of cumulative occurrences')
        plt.plot(ts, accumulations)
        plt.plot(ts, totals)
        plt.xlim(0.0, self.T)
        plt.show()

import numpy as np
import matplotlib.pyplot as plt


class HawkesProcess:
    """
    Hawkes過程
    Attribution:
        theta (np.arrray): Hawkes過程のパラメータ
    """

    def __init__(self) -> None:
        self.theta = np.array([0.01, 0.5, 0.5])

    def fit(self, t, T) -> None:
        """
        学習(負の対数尤度の最小化)
        Args:
            t: イベント発生時刻系列
            T: 観測期間
        """
        eta = 0.00001
        t = t[t < T]
        for i in range(1000):
            # パラメータ更新
            grad = self.nll_grad(t, T)
            self.theta = self.theta - eta * grad

    def nll_grad(self, t, T) -> np.array:
        """
        負の対数尤度の勾配計算
        Args:
            t: イベント発生時刻系列
            T: 観測期間
        """
        mu = self.theta[0]
        a = self.theta[1]
        b = self.theta[2]

        G = [0]
        G_db = [0]
        lam = [mu]
        lam_da = [0]
        lam_db = [0]
        # 漸化式による変数の事前計算
        for i in range(len(t) - 1):
            G.append((G[i] + a * b) * np.exp(-b * (t[i + 1] - t[i])))
            G_db.append((G_db[i] + a) * np.exp(-b * (t[i + 1] - t[i])) - G[i + 1] * (t[i + 1] - t[i]))
            lam.append(G[i + 1] + mu)
            lam_da.append(G[i + 1] / a)
            lam_db.append(G_db[i + 1])

        # 勾配の計算
        x1, x2, x3, x4, x5 = 0, 0, 0, 0, 0
        for i in range(len(t)):
            x1 += (1 / lam[i])
            x2 += (1 / lam[i]) * lam_da[i]
            x3 += 1 - np.exp(-b * (T - t[i]))
            x4 += (1 / lam[i]) * lam_db[i]
            x5 += a * (T - t[i]) * np.exp(-b * (T - t[i]))

        grad_mu = x1 - T
        grad_a = x2 - x3
        grad_b = x4 - x5
        return np.array([-grad_mu, -grad_a, -grad_b])

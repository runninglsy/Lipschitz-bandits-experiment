import numpy as np
import abc


class Algorithm(metaclass=abc.ABCMeta):  # abstract class of `Algorithm`
    def __init__(self, delta, T, c):
        self.delta = delta
        self.T = T
        self.c = c
        self.pulled_idx = None
        self.active_arms = None
        self.mu = None
        self.n = None
        self.r = None

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def output(self):
        pass

    @abc.abstractmethod
    def observe(self, t, y):
        pass

    def get_uncovered(self):
        covered = [[arm - r, arm + r] for arm, r in zip(self.active_arms, self.r)]
        if covered == []:
            return [0, 1]
        covered.sort(key=lambda x: x[0])
        low = 0
        for interval in covered:
            if interval[0] <= low:
                low = max(low, interval[1])
                if low >= 1:
                    return None
            else:
                return [low, interval[0]]
        return [low, 1]


class Zooming(Algorithm):
    def __init__(self, delta, T, c, nu):
        super().__init__(delta, T, c)
        self.nu = nu

    def initialize(self):
        self.active_arms = []
        self.mu = []
        self.n = []
        self.r = []

    def output(self):
        uncovered = self.get_uncovered()
        if uncovered is None:
            score = [mu + 2 * r for mu, r in zip(self.mu, self.r)]
            self.pulled_idx = np.argmax(score)
        else:
            new_arm = np.random.uniform(*uncovered)
            self.active_arms.append(new_arm)
            self.mu.append(0)
            self.n.append(0)
            self.r.append(0)
            self.pulled_idx = len(self.active_arms) - 1
        return self.pulled_idx

    def observe(self, t, y):
        idx = self.pulled_idx
        self.mu[idx] = (self.mu[idx] * self.n[idx] + y) / (self.n[idx] + 1)
        self.n[idx] += 1
        for i, n in enumerate(self.n):
            self.r[i] = self.c * self.nu * np.power(t, 1 / 3) / np.sqrt(n)


class ADTM(Algorithm):
    def __init__(self, delta, T, c, nu, epsilon):
        super().__init__(delta, T, c)
        self.nu = nu
        self.epsilon = epsilon

    def initialize(self):
        self.active_arms = []
        self.mu = []
        self.n = []
        self.r = []

    def output(self):
        uncovered = self.get_uncovered()
        if uncovered is None:
            score = [mu + 2 * r for mu, r in zip(self.mu, self.r)]
            self.pulled_idx = np.argmax(score)
        else:
            new_arm = np.random.uniform(*uncovered)
            self.active_arms.append(new_arm)
            self.mu.append(0)
            self.n.append(0)
            self.r.append(0)
            self.pulled_idx = len(self.active_arms) - 1
        return self.pulled_idx

    def observe(self, t, y):
        idx = self.pulled_idx
        threshold = np.power(self.nu * (self.n[idx] + 1) / np.log(self.T ** 2 / self.delta), 1 / (1 + self.epsilon))
        if abs(y) > threshold:
            y = 0
        self.mu[idx] = (self.mu[idx] * self.n[idx] + y) / (self.n[idx] + 1)
        self.n[idx] += 1
        self.r[idx] = self.c * 4 * np.power(self.nu, 1 / (1 + self.epsilon)) * np.power(
            np.log(self.T ** 2 / self.delta) / self.n[idx], self.epsilon / (1 + self.epsilon))


class ADMM(Algorithm):
    def __init__(self, delta, T, c, sigma, epsilon):
        super().__init__(delta, T, c)
        self.sigma = sigma
        self.epsilon = epsilon
        self.h = None
        self.replay = None

    def initialize(self):
        self.active_arms = []
        self.mu = []
        self.n = []
        self.r = []
        self.h = []
        self.replay = False

    def output(self):
        if self.replay:
            pass  # remain `self.pulled_idx` unchanged
        else:
            uncovered = self.get_uncovered()
            if uncovered is None:
                score = [mu + 2 * r for mu, r in zip(self.mu, self.r)]
                self.pulled_idx = np.argmax(score)
            else:
                new_arm = np.random.uniform(*uncovered)
                self.active_arms.append(new_arm)
                self.mu.append(0)
                self.h.append([])
                self.n.append(0)
                self.r.append(0)
                self.pulled_idx = len(self.active_arms) - 1
        return self.pulled_idx

    def observe(self, t, y):
        def MME(rewards):
            M = int(np.floor(8 * np.log(self.T ** 2 / self.delta) + 1))
            B = int(np.floor(len(rewards) / M))
            means = np.zeros(M)
            for m in range(M):
                means[m] = np.mean(rewards[m * B:(m + 1) * B])
            return np.median(means)

        idx = self.pulled_idx
        self.h[idx].append(y)
        self.n[idx] += 1
        self.r[idx] = self.c * np.power(12 * self.sigma, 1 / (1 + self.epsilon)) * np.power(
            (16 * np.log(self.T ** 2 / self.delta) + 2) / self.n[idx], self.epsilon / (1 + self.epsilon))
        if self.n[idx] < 16 * np.log(self.T ** 2 / self.delta) + 2:
            self.replay = True
        else:
            self.replay = False
            self.mu[idx] = MME(self.h[idx])

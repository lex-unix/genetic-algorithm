import numpy as np


class Easom:
    bounds = [[-100.0, 100.0], [-100.0, 100.0]]

    def func(self, x, y):
        return -1 * np.cos(x) * np.cos(y) * np.exp(-1 * ((x - np.pi)**2 + (y - np.pi)**2))


class ThreeHumpCamel:
    bounds = [[-5.0, 5.0], [-5.0, 5.0]]

    def func(self, x, y):
        return 2*x**2 - 1.05 * x**4 + x**6 / 6 + x*y + y**2


class Ackley:
    bounds = [[-5.0, 5.0], [-5.0, 5.0]]

    def func(self, x, y):
        return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

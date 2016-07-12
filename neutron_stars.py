#!/usr/local/bin/python3

import math
import numpy as np
import scipy
import scipy.integrate as integrate
import scipy.misc
import matplotlib.pyplot


def rk4(f, x0, y0, x_max, n):
    dx = np.empty(n + 1)
    dy = np.empty(n + 1)
    h = (x_max - x0) / float(n)

    dx[0] = x = x0
    dy[0] = y = y0
    for i in range(1, n+1):
        k1 = h * f(x, y)
        k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(x + h, y + k3)

        dx[i] = x = x0 + i * h
        dy[i] = y = y + (k1 + 2*k2 + 2*k3 + k4) / 6

    return dx, dy


def euler(f, x0, y0, x_max, n):
    dx = np.empty(n + 1)
    dy = np.empty(n + 1)
    h = (x_max - x0) / float(n)

    dx[0] = x = x0
    dy[0] = y = y0
    for i in range(n):
        dy[i+1] = dy[i] + (h) * f(y, i)
        y = dy[i]
        print(y)
        # print(dy[i])

    return dx, dy


def simple_dx_dt(x, t):
    """
    dx
    -- = -t
    dt
    """

    return -t


def exact_dx_dt(x, t):
    """
    x = e^(t)
    """
    return math.pow(math.e, x)


def solve_rk4_dx_dt():
    return rk4(exact_dx_dt, 0, 1, 10, 10)


def solve_euler_dx_dt():
    return euler(exact_dx_dt, 0, 1, 10, 10)


def solve_exact_dx_dt():
    y = np.empty(11)
    for t in range(11):
        y[t] = exact_dx_dt(t, y)
        # print(y[t])

    return y


if __name__ == '__main__':
    sol = solve_exact_dx_dt()
    rk4_sol = solve_rk4_dx_dt()
    matplotlib.pyplot.plot(sol, 'b', label='e^(t)')
    matplotlib.pyplot.plot(rk4_sol[1], 'r', label='rk4')
    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.xlabel('t')
    matplotlib.pyplot.grid()
    matplotlib.pyplot.show()

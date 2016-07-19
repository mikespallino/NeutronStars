#!/usr/local/bin/python3

import math
import matplotlib.pyplot


def rk4(f, x0, y0, x_max, n):
    """
    Runge-Kutta solver for OEDs
    :param f: callable function f with positional arguments x, y
    :param x0: initial t value
    :param y0: inital value of f evaluated at x0
    :param x_max: max value of x
    :param n: slices
    :return: vectors dx, dy containing the solution to the OEDs
    """
    dx = [0] * (n + 1)
    dy = [0] * (n + 1)
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


def rk4_coupled(f, g, t0, f_y0, g_y0, t_max, n):
    """
    Runge-Kutta solver for Coupled OEDs
    :param f: callable function f with positional arguments t, f, g
    :param g: callable function g with positional arguments t, f, g
    :param t0: initial t value
    :param f_y0: initial value of f evaluated at t0
    :param g_y0: initial value of g evaluated at t0
    :param t_max: max value of t
    :param n: slices
    :return: vectors dt, df, dg containing the solution to the coupled OEDs
    """
    dt = [0] * (n + 1)
    df = [0] * (n + 1)
    dg = [0] * (n + 1)
    h = (t_max - t0) / float(n)

    dt[0] = t = t0
    df[0] = f_i = f_y0
    dg[0] = g_i = g_y0
    for i in range(1, n+1):
        try:
            k1_f = f(t,         f_i,              g_i)
            k1_g = g(t,         f_i,              g_i)
            k2_f = f(t + 0.5*h, f_i + h*0.5*k1_f, g_i + h*0.5*k1_g)
            k2_g = g(t + 0.5*h, f_i + h*0.5*k1_f, g_i + h*0.5*k1_g)
            k3_f = f(t + 0.5*h, f_i + h*0.5*k2_f, g_i + h*0.5*k2_g)
            k3_g = g(t + 0.5*h, f_i + h*0.5*k2_f, g_i + h*0.5*k2_g)
            k4_f = f(t + h,     f_i + h*k3_f,     g_i + h*k3_g)
            k4_g = g(t + h,     f_i + h*k3_f,     g_i + h*k3_g)

            dt[i] = t = t0 + i * h
            df[i] = f_i = f_i + float(h/6) * (k1_f + 2*k2_f + 2*k3_f + k4_f)
            dg[i] = g_i = g_i + float(h/6) * (k1_g + 2*k2_g + 2*k3_g + k4_g)
        except ValueError as e:
            # INFO: This means we should stop calculating here
            # We should truncate the lists so that the graphs don't look weird.
            # Only graph up to the valid results
            dt = dt[:i]
            df = df[:i]
            dg = dg[:i]
            break

    return dt, df, dg

# INFO: Constants for the TOV
ALPHA = 5
BETA = 1
GAMMA = 5/3.0
R_MAX = 30
G = 6.67408 * 10**-11
r_small = 0.001
m_small = 3.3827159301471463 * 10**-12
p_small = 0.0008075639420379252


def tov_coupled(r, p, m):
    """
    dp/dr EQ(22)
    :param r: radius
    :param p: pressure
    :param m: mass
    :return: pressure
    """
    return -1 * (ALPHA * math.pow(p, float(1/GAMMA)) * m) / float(math.pow(r, 2))


def mass(r, p, m):
    """
    dM/dr EQ(25)
    :param r: radius
    :param p: pressure
    :param m: mass
    :return: mass
    """
    return BETA * math.pow(r, 2) * math.pow(p, float(1/GAMMA))


def solve_coupled_tov():
    """
    This function just calls rk4_coupled with the above functions and the initial values
    :return: solution to the TOV
    """
    return rk4_coupled(tov_coupled, mass, r_small, p_small, m_small, R_MAX, 1000000)


# INFO: TEST OF COUPLED RK4
a = 10
b = 1
l = 1
k = 1


def R(t, r, f):
    return a*r-b*r*f


def F(t, r, f):
    return -l*f+k*r*f


def solve_predator():
    return rk4_coupled(R, F, 0, 20, 0, 5, 50, 100000)


if __name__ == '__main__':
    # INFO: Test code validating rk4_coupled
    # rk4_sol = solve_predator()
    # matplotlib.pyplot.plot(rk4_sol[0], rk4_sol[2], 'b', label='rabbits')
    # matplotlib.pyplot.plot(rk4_sol[0], rk4_sol[1], 'r', label='foxes')

    # INFO: Graph the solution for pressure
    rk4_sol = solve_coupled_tov()
    print(len(rk4_sol[0]))
    print(rk4_sol[0][len(rk4_sol[0])-1])
    matplotlib.pyplot.plot(rk4_sol[0], rk4_sol[1], 'b', label='dp/dr')
    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.xlabel('r')
    matplotlib.pyplot.grid()
    matplotlib.pyplot.show()

    # INFO: Graph the solution for mass
    matplotlib.pyplot.plot(rk4_sol[0], rk4_sol[2], 'r', label='dM/dr')
    matplotlib.pyplot.legend(loc='best')
    matplotlib.pyplot.xlabel('r')
    matplotlib.pyplot.grid()
    matplotlib.pyplot.show()

#!/usr/local/bin/python3

import math
import numpy as np
import scipy
import scipy.integrate as integrate
import scipy.misc
import matplotlib.pyplot

# Speed of light
c = float(3.0 * pow(10, 8))
# Newton's Gravitational constant
G = float(6.673 * pow(10, -8))
# Neutron Mass
M_a = float(1.674929 * pow(10,-27)) # kg
# Number of nucleons per electron
A_Z = 2.00
# Plank's constant
h_bar = float(1.0545718 * pow(10, -34)) # m2 kg / s
# Electron mass
m_e = float(9.10938356 * pow(10, -31)) # kg

# User defined kF
k_F = m_e

def newtonian_tov(r):
	return -(G * energy_density(r) * total_mass(r)) / (float(pow(c, 2)) * float(pow(r, 2)))


def free_electrons():
	return (pow(k_F, 3)) / (3 * pow(math.pi, 2) * pow(h_bar, 3))


def mass_density():
	return free_electrons() * M_a * A_Z


def energy_density(r):
	return mass_density() * float(pow(c, 2))


def total_mass_integral(r):
	"""
	NOTE: Only for use when integrating for total mass below.
	"""
	return (float(pow(r, 2)) * energy_density(r)) / float(pow(c, 2))


def total_mass(r):
	return 4 * math.pi * integrate.quad(total_mass_integral, 0, r)[1]


def rk4(func, max_x=10, step_size=1):
	# Step size
	h = step_size
	t = 0.001
	y = 0

	results = np.empty(max_x * (1/step_size))
	index = 0

	while t < max_x:
		# K values for rk4
		k1 = func(t)
		k2 = func(t + (h/2)* k1)
		k3 = func(t + (h/2)* k2)
		k4 = func(t + h*k3)

		y += float((h/6) * float(k1 + 2*k2 + 2*k3 + k4))
		results[index] = y
		index += 1
		t += h

	return results


if __name__ == '__main__':
	print("Calculating...")
	res = rk4(newtonian_tov, max_x=30, step_size=0.001)
	print("Plotting...")
	matplotlib.pyplot.figure("Newtonian TOV")
	matplotlib.pyplot.plot(res)
	matplotlib.pyplot.show() 
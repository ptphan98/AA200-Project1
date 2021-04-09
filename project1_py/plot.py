import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from optimizers import *
import helpers

# plot Rosenbrock Cases
Ros= helpers.Simple1()

x1 = np.array([0.5, -1.9])
x2 = np.array([1.5, .5])
x3 = np.array([-1.8, 0.6])


# plot meshgrid
delta = 0.1
x = np.arange(-2.0, 2.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
values = np.array([X, Y])
Z = Ros.f(values)

Ros._reset()
X1_vals = Adam(Ros.f, Ros.g, x1, Ros.n, Ros.count, Ros.prob)

Ros._reset()
X2_vals = Adam(Ros.f, Ros.g, x2, Ros.n, Ros.count, Ros.prob)

Ros._reset()
X3_vals = Adam(Ros.f, Ros.g, x3, Ros.n, Ros.count, Ros.prob)

fig, ax = plt.subplots()
levels = np.arange(0, 3500, 50)
CS = ax.contour(X, Y, Z, levels)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Rosenbrock’s Function Optimization Using Adam (n=10)')
ax.plot(X1_vals[:, 0], X1_vals[:, 1], 'b')
ax.plot(X2_vals[:, 0], X2_vals[:, 1], 'g')
ax.plot(X3_vals[:, 0], X3_vals[:, 1], 'r')
ax.plot(X1_vals[0, 0], X1_vals[0, 1], 'b*')
ax.plot(X2_vals[0, 0], X2_vals[0, 1], 'g*')
ax.plot(X3_vals[0, 0], X3_vals[0, 1], 'r*')

plt.show()

# plot convergence functions

# Rosenbrock’s Function
X1_vals = np.array([X1_vals[:, 0], X1_vals[:, 1]])
X2_vals = np.array([X2_vals[:, 0], X2_vals[:, 1]])
X3_vals = np.array([X3_vals[:, 0], X3_vals[:, 1]])

Fun_val1 = Ros.f(X1_vals)
Fun_val2 = Ros.f(X2_vals)
Fun_val3 = Ros.f(X3_vals)
steps = np.arange(0, Fun_val1.shape[0])

fig, ax = plt.subplots()
ax.plot(steps, Fun_val1,'b')
ax.plot(steps, Fun_val2,'g')
ax.plot(steps, Fun_val3,'r')
ax.set(xlabel='Iteration', ylabel='Function Value',
       title='Rosenbrock’s Function Convergence (Adam)')

# Himmelblau’s Function
Him = helpers.Simple2()

Him._reset()
X1_vals = RMSProp(Him.f, Him.g, x1, Him.n, Him.count, Him.prob)
Him._reset()
X2_vals = RMSProp(Him.f, Him.g, x2, Him.n, Him.count, Him.prob)
Him._reset()
X3_vals = RMSProp(Him.f, Him.g, x3, Him.n, Him.count, Him.prob)

X1_vals = np.array([X1_vals[:, 0], X1_vals[:, 1]])
X2_vals = np.array([X2_vals[:, 0], X2_vals[:, 1] ])
X3_vals = np.array([X3_vals[:, 0], X3_vals[:, 1]])

Fun_val1 = Him.f(X1_vals)
Fun_val2 = Him.f(X2_vals)
Fun_val3 = Him.f(X3_vals)
steps = np.arange(0, Fun_val1.shape[0])

fig, ax = plt.subplots()
ax.plot(steps, Fun_val1,'b')
ax.plot(steps, Fun_val2,'g')
ax.plot(steps, Fun_val3,'r')
ax.set(xlabel='Iteration', ylabel='Function Value',
       title='Himmelblau’s Function Convergence (RMSProp)')

# Powell’s function
Pow = helpers.Simple3()
x1 = [ 0.62359889, -1.61698917, -0.68876633, -0.4567085 ]
x2 = [-0.02985691, -2.21944699,  1.34012007, 0.2193125 ]
x3 = [ 1.799162, -2.04004373, -2.00125211, 0.1651603 ]

print(x1)
print(x2)
print(x3)

Pow._reset()
X1_vals = RMSProp(Pow.f, Pow.g, x1, Pow.n, Pow.count, Pow.prob)
Pow._reset()
X2_vals = RMSProp(Pow.f, Pow.g, x2, Pow.n, Pow.count, Pow.prob)
Pow._reset()
X3_vals = RMSProp(Pow.f, Pow.g, x3, Pow.n, Pow.count, Pow.prob)

X1_vals = np.array([X1_vals[:, 0], X1_vals[:, 1], X1_vals[:, 2], X1_vals[:, 3]])
X2_vals = np.array([X2_vals[:, 0], X2_vals[:, 1], X2_vals[:, 2], X2_vals[:, 3] ])
X3_vals = np.array([X3_vals[:, 0], X3_vals[:, 1], X3_vals[:, 2], X3_vals[:, 3]])

Fun_val1 = Pow.f(X1_vals)
Fun_val2 = Pow.f(X2_vals)
Fun_val3 = Pow.f(X3_vals)
steps = np.arange(0, Fun_val1.shape[0])

fig, ax = plt.subplots()
ax.plot(steps, Fun_val1,'b')
ax.plot(steps, Fun_val2,'g')
ax.plot(steps, Fun_val3,'r')
ax.set(xlabel='Iteration', ylabel='Function Value',
       title='Powell’s Function Convergence (RMSProp)')
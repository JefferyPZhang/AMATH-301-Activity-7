import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

# Set Up

def forward_euler(f, t_span, y_0):
    sol = np.zeros(len(t_span))
    sol[0] = y_0
    step = t_span[1] - t_span[0]
    for i in range (1, len(t_span)):
        sol[i] = sol[i - 1] + step * f(t_span[i - 1], sol[i - 1])
    return t_span, sol
    
def backward_euler(f, t_span, y_0):
    sol = np.zeros(len(t_span))
    sol[0] = y_0
    step = t_span[1] - t_span[0]
    for i in range (1, len(t_span)):
        g = lambda x: x - sol[i - 1] - step * f(t_span[i], x)
        sol[i] = scipy.optimize.fsolve(g, sol[i - 1])[0]
    return t_span, sol

def RK2(f, t_span, y_0):
    sol = np.zeros(len(t_span))
    sol[0] = y_0
    step = t_span[1] - t_span[0]
    for i in range (1, len(t_span)):
        sol[i] = sol[i - 1] + (step * f(t_span[i - 1] + (step / 2), sol[i - 1] + ((step / 2) * f(t_span[i - 1], sol[i - 1]))))
    return t_span, sol
    
dydt = lambda t, y: 0.3 * y + t
ts, ys = forward_euler(dydt, np.arange(0, 1.1, 0.1), 1)
A1 = ys[-1]
ts, ys = backward_euler(dydt, np.arange(0, 1.1, 0.1), 1)
A2 = ys[-1]
ts, ys = RK2(dydt, np.arange(0, 1.1, 0.1), 1)
A3 = ys[-1]
ys = scipy.integrate.solve_ivp(dydt, [0, 1], [1])
A4 = ys.y[0][-1]

# System 1

dydt = lambda t, y: y ** 3 - y
exact_sol = 3.451397662017099
t_span = np.arange(0, 0.1 + 0.001, 0.001)

ts, ys = forward_euler(dydt, t_span, 2) # shortest
A5 = np.abs(ys[-1] - exact_sol)
ts, ys = backward_euler(dydt, t_span, 2) # longest, largest error
A6 = np.abs(ys[-1] - exact_sol)
ts, ys = RK2(dydt, t_span, 2)
A7 = np.abs(ys[-1] - exact_sol)
ys = scipy.integrate.solve_ivp(dydt, [0, 0.1], [2])
A8 = np.abs(ys.y[0][-1] - exact_sol) # smallest error
A9 = [4, 3, 1, 2]
A10 = [1, 3, 4, 2]
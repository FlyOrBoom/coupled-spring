from numpy import sin, cos, pi, sqrt
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

l = 1.0  # initial length of spring in m
phi = 1.0 # initial winding angle of spring in rad
r = 0.1  # radius of spring in m
R = 0.3 # radius of ring

k = 20.0 # linear spring constant in N/m
kappa = 10.0 # torsional spring constant in N*m/rad

M = 1.0  # mass of ring in kg
I = M*R*R # moment of inertia of ring in kg*m^2

t_stop = 10  # how many seconds to simulate
history_len = 500  # how many trajectory points to display

X, V, THETA, OMEGA = 0,1,2,3

x0 = -0.05*l # initial vertical displacement in m
v0 = 0.0*l # initial vertical velocity in m/s
theta0 = 0.05*phi # initial anglular displacement in rad
omega0 = -0.0*phi # initial angular velocity in rad/s

# initial state
state = [x0, v0, theta0, omega0]

def derivs(t, state):
    ddt = np.zeros_like(state)

    x = state[X]
    v = state[V]
    theta = state[THETA]
    omega = state[OMEGA]

    a = v*v + r*r*omega*omega - (k*x/M) - (r*r*kappa*theta/I)
    b = (v*(x+l)/M) + (r*r*r*r*theta*(phi+theta)/I)
    epsilon = 1e-3
    lamb = (a*b)/(b*b + epsilon*epsilon)

    ddt[X] = v
    ddt[V] = -(k*x + lamb*v*(x+l))/M

    ddt[THETA] = omega
    ddt[OMEGA] = -(kappa*theta + lamb*r*r*omega*(phi+theta))/I

    return ddt

# create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.002
t = np.arange(0, t_stop, dt)

# integrate the ODE using Euler's method
y = np.empty((len(t), 4))
y[0] = state
#for i in range(1, len(t)):
    #y[i] = y[i - 1] + derivs(t[i - 1], y[i - 1]) * dt

# A more accurate estimate could be obtained e.g. using scipy:
solve = scipy.integrate.solve_ivp(derivs, t[[0, -1]], state, t_eval=t)
print(solve)
y = solve.y.T

x = y[:, X]
v = y[:, V]
theta = y[:, THETA]
omega = y[:, OMEGA]

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-l/10, l/10), ylim=(-phi/10, phi/10))
ax.set_xlabel("Linear displacement (m)")
ax.set_ylabel("Angular displacement (rad)")
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)


def animate(i):
    thisx = [0, x[i]]#, v[i]]
    thisy = [0, theta[i]]#, omega[i]]

    if i == 0:
        history_x.clear()
        history_y.clear()

    history_x.appendleft(thisx[1])
    history_y.appendleft(thisy[1])

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return line, trace, time_text


ani = animation.FuncAnimation(
    fig, animate, len(y), interval=1, blit=True)
plt.show()

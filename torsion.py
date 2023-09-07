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

k = 10.0 # linear spring constant in N/m
kappa = 10.0 # torsional spring constant in N*m/rad

M = 1.0  # mass of ring in kg
I = M*R*R # moment of inertia of ring in kg*m^2

t_stop = 30  # how many seconds to simulate
history_len = 1000  # how many trajectory points to display

X, V, THETA, OMEGA = 0,1,2,3

x0 = -0.04*l # initial vertical displacement in m
v0 = 0.02*l # initial vertical velocity in m/s
theta0 = -0.05*phi # initial anglular displacement in rad
omega0 = -0.02*phi # initial angular velocity in rad/s

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
speed = 4
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

fig, (delta_axes, net_axes) = plt.subplots(1, 2)

delta_axes.set_ylim(-l*0.1, l*0.1)
delta_axes.set_ylabel("Linear displacement (m)")
delta_axes.set_xlim(-phi*0.1, phi*0.1)
delta_axes.set_xlabel("Angular displacement (rad)")
delta_axes.invert_yaxis()
delta_axes.grid()

net_axes.set_ylim(0, l*1.1)
net_axes.set_ylabel("Axial position (m)")
net_axes.set_xlim(-r*1.1, r*1.1)
net_axes.set_xlabel("Transverse position (m)")
net_axes.invert_yaxis()
net_axes.grid()

time_template = 'time = %.1fs'
time_text = delta_axes.text(0.05, 0.9, '', transform=delta_axes.transAxes)

delta_line, = delta_axes.plot([], [], 'o-', lw=1)
delta_trace, = delta_axes.plot([], [], '.-', lw=1, ms=2)

net_line, = net_axes.plot([], [], '-x', lw=1)
net_ring, = net_axes.plot([], [], '-o', lw=1)

history_x, history_theta = deque(maxlen=history_len), deque(maxlen=history_len)

def animate(i):
    j = i*speed

    x_j = x[j]
    theta_j = theta[j]
    l_j = l + x_j

    if i == 0:
        history_x.clear()
        history_theta.clear()

    history_x.appendleft(x_j)
    history_theta.appendleft(theta_j)

    delta_line.set_data([0, theta_j], [0, x_j])
    delta_trace.set_data(history_theta, history_x)

    net_line.set_data([0, 0], [0, l_j])
    N = 36
    net_ring.set_data(
        [ r*sin(theta_j + 2*pi*n/N) for n in range(0,N+1) ],
        [ l_j + 0.1*r*cos(theta_j + 2*pi*n/N) for n in range(0,N+1) ]
    )

    time_text.set_text(time_template % (j*dt))

    return delta_line, delta_trace, net_line, net_ring, time_text

ani = animation.FuncAnimation(
    fig, animate, round(len(y)/speed), interval=1, blit=True)
plt.show()

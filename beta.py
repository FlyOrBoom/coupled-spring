from numpy import sin, cos, pi, arctan, sqrt
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

l = 1.0  # initial length of spring in m
φ = 20.0 # initial winding angle of spring in rad
r = 0.1  # radius of spring in m
R = 0.3 # radius of ring

k = 15.0 # linear spring constant in N/m
κ = 10.0 # torsional spring constant in N*m/rad

m = 1.0  # mass of ring in kg
I = m*R*R # moment of inertia of ring in kg*m^2

t_stop = 10  # how many seconds to simulate
history_len = 1000  # how many trajectory points to display

γ = 1.0 # coupling factor; should be 1

d = sqrt(l*l + r*r*φ*φ)
b0 = arctan(l/r/φ)
bb0 = 0.3

# initial state
state = [b0, bb0]

def derivs(_, state):
    ddt = np.zeros_like(state)

    b = state[0]
    bb = state[1]

    numer = -m*bb*bb*sin(2*b) - I*bb*bb*sin(2*b)/r/r + 2*k*(sin(b) - sin(b0))*cos(b) - 2*k*(cos(b)-cos(b0))*sin(b)/r/r
    denom = -(2*m*cos(b)*cos(b) + 2*I*sin(b)*sin(b)/r/r)

    ddt[0] = bb
    ddt[1] = numer/denom

    return ddt

# create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.001
speed = 10
t = np.arange(0, t_stop, dt)

# integrate the ODE using Euler's method
y = np.empty((len(t), 2))
y[0] = state
solve = scipy.integrate.solve_ivp(derivs, t[[0, -1]], state, t_eval=t)

print(solve)
y = solve.y.T

array_b = y[:, 0]
array_bb = y[:, 1]

fig, (delta_axes, net_axes, energy_axes) = plt.subplots(1, 3)

delta_axes.set_ylim(-l*0.1, l*0.1)
delta_axes.set_ylabel("Linear displacement (m)")
delta_axes.set_xlim(-φ*0.1, φ*0.1)
delta_axes.set_xlabel("Angular displacement (rad)")
delta_axes.invert_yaxis()
delta_axes.grid()

net_axes.set_ylim(0, l*1.1)
net_axes.set_ylabel("Axial position (m)")
net_axes.set_xlim(-R*1.1, R*1.1)
net_axes.set_xlabel("Transverse position (m)")
net_axes.invert_yaxis()
net_axes.grid()

energy_axes.set_ylim(0, t_stop)
energy_axes.set_ylabel("Time (s)")
energy_axes.set_xlim(0, 3)
energy_axes.set_xlabel("Energy (J)")
energy_axes.invert_yaxis()
energy_axes.grid()

time_template = 'time = %.1fs'
time_text = delta_axes.text(0.05, 0.9, '', transform=delta_axes.transAxes)

delta_line, = delta_axes.plot([], [], 'o-', lw=1)
delta_trace, = delta_axes.plot([], [], '.-', lw=1, ms=2)

net_line, = net_axes.plot([], [], '-', lw=1)
net_ring, = net_axes.plot([], [], '-o', lw=1)

energy_x, = energy_axes.plot([], [], '-', lw=1)
energy_v, = energy_axes.plot([], [], '-', lw=1)
energy_θ, = energy_axes.plot([], [], '-', lw=1)
energy_ω, = energy_axes.plot([], [], '-', lw=1)
energy_net, = energy_axes.plot([], [], '-', lw=1)

history_t = deque(maxlen=history_len)
history_x, history_θ = deque(maxlen=history_len), deque(maxlen=history_len)
history_Ex, history_Eθ = deque(maxlen=history_len), deque(maxlen=history_len)
history_Ev, history_Eω = deque(maxlen=history_len), deque(maxlen=history_len)
history_Enet = deque(maxlen=history_len)

def animate(i):
    j = i*speed
    t = j*dt

    b = array_b[j]
    bb = array_bb[j]

    x = d*(sin(b) - sin(b0))
    v = d*bb*cos(b)
    θ = d*(cos(b) - cos(b0))/r
    ω = d*bb*sin(b)/r

    Ex = k*x*x/2
    Ev = m*v*v/2
    Eθ = κ*θ*θ/2
    Eω = I*ω*ω/2

    if i == 0:
        history_t.clear()
        history_x.clear()
        history_θ.clear()

        history_Ex.clear()
        history_Eθ.clear()
        history_Ev.clear()
        history_Eω.clear()
        history_Enet.clear()

    history_t.append(t)
    history_x.append(x)
    history_θ.append(θ)

    history_Ex.append(Ex)
    history_Eθ.append(Eθ)
    history_Ev.append(Ev)
    history_Eω.append(Eω)
    history_Enet.append(Ex+Eθ+Ev+Eω)

    delta_line.set_data([0, θ], [0, x])
    delta_trace.set_data(history_θ, history_x)

    N_arcs = 500
    net_line.set_data(
        [r*sin((φ+θ)*n/N_arcs) for n in range(N_arcs+1)], 
        [(n/N_arcs) * (l+x) for n in range(N_arcs+1)]
    )
    N_masses = 17
    net_ring.set_data(
        [ R*sin(θ + 2*pi*n/N_masses) for n in range(N_masses+1) ],
        [ l+x + 0.1*R*cos(θ + 2*pi*n/N_masses) for n in range(N_masses+1) ]
    )

    energy_x.set_data(history_Ex, history_t)
    energy_v.set_data(history_Ev, history_t)
    energy_θ.set_data(history_Eθ, history_t)
    energy_ω.set_data(history_Eω, history_t)
    energy_net.set_data(history_Enet, history_t)

    time_text.set_text(time_template % t)

    return delta_line, delta_trace, net_line, net_ring, energy_x, energy_v, energy_θ, energy_ω, energy_net, time_text

ani = animation.FuncAnimation(
    fig, animate, round(len(y)/speed), interval=1, blit=True)
plt.show()

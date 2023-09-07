from numpy import sin, cos, pi, sqrt
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

l = 1.0  # initial length of spring in m
φ = 1.0 # initial winding angle of spring in rad
r = 0.1  # radius of spring in m
R = 0.3 # radius of ring

k = 10.0 # linear spring constant in N/m
κ = 10.0 # torsional spring constant in N*m/rad

M = 1.0  # mass of ring in kg
I = M*R*R # moment of inertia of ring in kg*m^2

t_stop = 30  # how many seconds to simulate
history_len = 1000  # how many trajectory points to display

γ = 1.0 # coupling constant; should be 1

X, V, Θ, Ω = 0,1,2,3

x0 = -0.04*l # initial vertical displacement in m
v0 = 0.02*l # initial vertical velocity in m/s
θ0 = -0.05*φ # initial anglular displacement in rad
ω0 = 0.02*φ # initial angular velocity in rad/s

# initial state
state = [x0, v0, θ0, ω0]

def derivs(_, state):
    ddt = np.zeros_like(state)

    x = state[X]
    v = state[V]
    θ = state[Θ]
    ω = state[Ω]

    a = v*v + r*r*ω*ω - (k*x*(x+l)/M) - (r*r*κ*θ*(φ+θ)/I)
    b = (v*(x+l)*(x+l)/M) + (r*r*r*r*ω*(φ+θ)*(φ+θ)/I)
    ε = 1e-5
    λ = (a*b)/(b*b + ε*ε)

    ddt[X] = v
    ddt[V] = -(k*x + γ*λ*v*(x+l))/M

    ddt[Θ] = ω
    ddt[Ω] = -(κ*θ + γ*λ*r*r*ω*(φ+θ))/I

    return ddt

# create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.001
speed = 10
t = np.arange(0, t_stop, dt)

# integrate the ODE using Euler's method
y = np.empty((len(t), 4))
y[0] = state
solve = scipy.integrate.solve_ivp(derivs, t[[0, -1]], state, t_eval=t)

print(solve)
y = solve.y.T

array_x = y[:, X]
array_v = y[:, V]
array_θ = y[:, Θ]
array_ω = y[:, Ω]

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
energy_axes.set_xlim(0, (k*x0*x0 + κ*θ0*θ0 + M*v0*v0 + I*ω0*ω0)/2)
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

history_t = deque(maxlen=history_len)
history_x, history_θ = deque(maxlen=history_len), deque(maxlen=history_len)
history_Ex, history_Eθ = deque(maxlen=history_len), deque(maxlen=history_len)
history_Ev, history_Eω = deque(maxlen=history_len), deque(maxlen=history_len)

def animate(i):
    j = i*speed
    t = j*dt

    x = array_x[j]
    v = array_v[j]
    θ = array_θ[j]
    ω = array_ω[j]

    Ex = k*x*x/2
    Ev = M*v*v/2
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

    history_t.append(t)
    history_x.append(x)
    history_θ.append(θ)

    history_Ex.append(Ex)
    history_Eθ.append(Eθ)
    history_Ev.append(Ev)
    history_Eω.append(Eω)

    delta_line.set_data([0, θ], [0, x])
    delta_trace.set_data(history_θ, history_x)

    N_arcs = 500
    net_line.set_data(
        [r*sin((φ+θ)*n/5) for n in range(N_arcs+1)], 
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

    time_text.set_text(time_template % t)

    return delta_line, delta_trace, net_line, net_ring, energy_x, energy_v, energy_θ, energy_ω, time_text

ani = animation.FuncAnimation(
    fig, animate, round(len(y)/speed), interval=1, blit=True)
plt.show()

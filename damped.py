import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

xi = 1 # initial displacement (m)
vi = 0 # initial velocity (m/s)
ti = 0 # initial time (s)
tf = 4 # final time (s)
samples = 1000 # time resolution (Hz)
array_t = np.linspace(ti, tf, num=samples) # array of time values

fig = plt.figure() # matplotlib figure
ax = fig.add_subplot(
    xlim = (ti, tf), xlabel = "Time (s)",
    ylim = (-xi, +xi), ylabel = "Linear displacement (m)"
) # main plot
fig.subplots_adjust(bottom=0.3) # shift main plot up

ax_beta = fig.add_axes([0.25, 0.15, 0.65, 0.03]) # beta slider
slider_beta = Slider(
    ax=ax_beta, label='beta (rad²/s²)',
    valmin=0, valmax=10, valinit=3
)

ax_omega0 = fig.add_axes([0.25, 0.05, 0.65, 0.03]) # omega0 slider
slider_omega0 = Slider(
    ax=ax_omega0, label='omega0 (rad/s)',
    valmin=0, valmax=100, valinit=30
)

def ddt(_, state): # [x, v] -> [dx/dt, dv/dt]
    [x, v] = state
    beta = slider_beta.val
    omega0 = slider_omega0.val

    return [ v, -(2*beta*v + omega0*omega0*x) ]

def array_x(): # array of displacement values
    state = [xi, vi]
    solve = scipy.integrate.solve_ivp(ddt, [ti, tf], state, t_eval=array_t)
    return solve.y.T[:, 0]

line, = ax.plot(array_t, np.zeros_like(array_t), '-', lw=1)

def update(_): # update values on slider input
    line.set_ydata(array_x())
    fig.canvas.draw_idle()

update(0)

# register the update function with each slider
slider_omega0.on_changed(update)
slider_beta.on_changed(update)

plt.show()

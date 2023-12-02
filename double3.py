import scipy
import scipy.integrate
import numpy as np
from numpy import pi, sin, cos, round
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

m1 = 0.3 # kg
m2 = 0.5 # kg
L1 = 0.5 # m 
L2 = 1.0 # m 
theta1i = 2.02 # rad
theta2i = 1 # rad
g = 9.8 # N/m

r = (L1+L2)*1.1

ti = 0.0 # s
tf = 10.0 # s 

samples = 10000 # time resolution (Hz)
dt = (tf-ti)/samples # interval (s)
arrays = {
    "t": np.linspace(ti, tf, num=samples)
}

slider_dims = np.array([0.25, 0.05, 0.65, 0.03])
dy = np.array([0, 0.04, 0, 0])

fig = plt.figure() # matplotlib figure
axes = {
    # physical
    "r": fig.add_subplot(
        2, 2, 1,
        xlim = (-r, +r), xlabel = "x displacement (m)",
        ylim = (-r, +r), ylabel = "y displacement (m)",
        aspect = "equal",
    ),
    # phase
    "theta": fig.add_subplot(
        2, 2, 2,
        xlim = (-pi, +pi), xlabel = "angle 1 (rad)",
        ylim = (-pi, +pi), ylabel = "angle 2 (rad)",
        aspect = "equal",
    ),
    # energies
    "energies": fig.add_subplot(
        2, 2, 3, 
        xlim = (ti,tf), xlabel = "time (s)",
        ylim = ( -g*(m1*L1 + m2*(L1+L2)), g*(m1*L1 + m2*(L1+L2))), ylabel = "energy (J)",
    ),
    # lagrangian
    "lagrange": fig.add_subplot(
        2, 2, 4, 
        xlim = (ti,tf), xlabel = "time (s)",
        ylim = ( 0, 2*g*(m1*L1 + m2*(L1+L2))), ylabel = "lagrangian (J)",
    ),
    # sliders
    "m1": fig.add_axes(slider_dims+5*dy),
    "m2": fig.add_axes(slider_dims+4*dy),
    "L1": fig.add_axes(slider_dims+3*dy),
    "L2": fig.add_axes(slider_dims+2*dy),
    "theta1i": fig.add_axes(slider_dims+1*dy),
    "theta2i": fig.add_axes(slider_dims+0*dy),
}
fig.subplots_adjust(bottom=0.4) # shift main plots up

sliders = {
    "m1": Slider( ax=axes["m1"], label="mass 1 (kg)", valmin=0.01, valmax=1, valinit=m1),
    "m2": Slider( ax=axes["m2"], label="mass 2 (kg)", valmin=0.01, valmax=1, valinit=m2),
    "L1": Slider( ax=axes["L1"], label="length 1 (m)", valmin=0.01, valmax=1, valinit=L1),
    "L2": Slider( ax=axes["L2"], label="length 2 (m)", valmin=0.01, valmax=1, valinit=L2),
    "theta1i": Slider( ax=axes["theta1i"], label="initial angle 1 (rad)", valmin=-pi, valmax=+pi, valinit=theta1i),
    "theta2i": Slider( ax=axes["theta2i"], label="initial angle 2 (rad)", valmin=-pi, valmax=+pi, valinit=theta2i),
}

def ddt(_, state): # [x, v] -> [dx/dt, dv/dt]
    [theta1, theta2, omega1, omega2] = state

    numer1 = -g*(2*m1 + m2)*sin(theta1) - m2*g*sin(theta1-2*theta2) - 2*sin(theta1-theta2)*m2*(omega2*omega2*L2 + omega1*omega1*L1*cos(theta1-theta2))

    numer2 = 2*sin(theta1-theta2) * ( (m1+m2)*L1*omega1*omega1 + (m1+m2)*g*cos(theta1) + m2*L2*omega2*omega2*cos(theta1-theta2) )

    denom = 2*m1 + m2 - m2*cos(2*(theta1-theta2))

    alpha1 = numer1/L1/denom
    alpha2 = numer2/L2/denom

    return [ omega1, omega2, alpha1, alpha2 ]

def solve(): # array of displacement values
    state = [theta1i, theta2i, 0, 0]
    solve = scipy.integrate.solve_ivp(ddt, [ti, tf], state, t_eval=arrays["t"]).y.T

    theta1s = solve[:, 0]
    theta2s = solve[:, 1]
    
    omega1s = solve[:, 2]
    omega2s = solve[:, 3]
    
    tee = (0.5*(m1+m2)*(L1**2)*(omega1s**2)) + (0.5*m2*(L2**2)*(omega2s**2)) + (m2*L1*L2*omega1s*omega2s*cos(theta1s-theta2s)) 
    vee = (-1*(m1+m2)*g*L1*cos(theta1s)) + (-1*m2*g*L2*cos(theta2s))
    lag = tee - vee
    ham = tee + vee
    
    arrays["theta"] = (np.array([theta1s, theta2s ]) + pi) % (2*pi) - pi
    arrays["path1"] = L1 * np.array([ sin(theta1s), -cos(theta1s) ])
    arrays["path2"] = np.sum([
        arrays["path1"], 
        L2 * np.array([ sin(theta2s), -cos(theta2s) ])
    ], axis=0)
    arrays["lagrangian"] =  np.array([(arrays["t"]),lag])
    arrays["hamiltonian"] = np.array([(arrays["t"]),ham])
    arrays["kineticenergy"] = np.array([(arrays["t"]),tee])
    arrays["potentialenergy"] = np.array([(arrays["t"]),vee])
   
    lines["theta"].set_data(*arrays["theta"])
    lines["path1"].set_data(*arrays["path1"])
    lines["path2"].set_data(*arrays["path2"])
    lines["lagrangian"].set_data(*arrays["lagrangian"])
    lines["hamiltonian"].set_data(*arrays["hamiltonian"])
    lines["kineticenergy"].set_data(*arrays["kineticenergy"])
    lines["potentialenergy"].set_data(*arrays["potentialenergy"])
    
lines = {
    "path1": axes["r"].plot([], [], '.', ms=1, color="pink")[0],
    "path2": axes["r"].plot([], [], '.', ms=1, color="lightblue")[0],
    "theta": axes["theta"].plot([], [], '.', ms=1, color="gray")[0],
    "lagrangian": axes["lagrange"].plot([],[],'.',ms=1, color = "green")[0],
    "hamiltonian": axes["energies"].plot([],[],'.',ms=1,color = "red")[0],
    "kineticenergy": axes["energies"].plot([],[],'.',ms=1, color = "purple")[0],
    "potentialenergy": axes["energies"].plot([],[],'.',ms=1, color = "black")[0],

    "path1_clip": axes["r"].plot([], [], '.', ms=2, color="red")[0],
    "path2_clip": axes["r"].plot([], [], '.', ms=2, color="blue")[0],
    "theta_clip": axes["theta"].plot([], [], '.', ms=2, color="black")[0],

    "rod1": axes["r"].plot([], [], 'o-', lw=1, color="red", markevery=[1])[0],
    "rod2": axes["r"].plot([], [], 'o-', lw=1, color="blue", markevery=[1])[0],
    "rodtheta": axes["theta"].plot([], [], 'o-', lw=1, color="black", markevery=[1])[0],

    "time": axes["r"].text(0.05, 0.9, "", transform=axes["r"].transAxes)
}

def animate(frame):
    lookback = max(0, frame-200)

    lines["rod1"].set_data(*np.array([(0,0), arrays["path1"][:, frame]]).T)
    lines["rod2"].set_data(*np.array([arrays["path1"][:, frame], arrays["path2"][:, frame]]).T)
    lines["rodtheta"].set_data(*np.array([(0,0), arrays["theta"][:, frame]]).T)

    lines["theta_clip"].set_data(*arrays["theta"][:, lookback:frame])
    lines["path1_clip"].set_data(*arrays["path1"][:, lookback:frame])
    lines["path2_clip"].set_data(*arrays["path2"][:, lookback:frame])

    lines["time"].set_text(f"t = {round(frame*dt, decimals=2)} s")

    return tuple(lines.values())

init = True

def update(_): # update values on slider input
    global init, m1, m2, L1, L2, theta1i, theta2i

    m1 = sliders["m1"].val
    m2 = sliders["m2"].val

    L1 = sliders["L1"].val
    L2 = sliders["L2"].val
    r = (L1+L2)*1.1
    axes["r"].set_xlim(-r, +r)
    axes["r"].set_ylim(-r, +r)

    theta1i = sliders["theta1i"].val
    theta2i = sliders["theta2i"].val

    if not init: ani.pause()

    solve()

    if not init: ani.resume()
    else: init = False

# register the update function with each slider
for slider in sliders.values():
    slider.on_changed(update)

update(0)

ani = animation.FuncAnimation(fig=fig, func=animate, frames=samples, interval=1000*dt, blit=True)
plt.show()


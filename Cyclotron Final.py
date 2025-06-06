#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:30:58 2024

@author: leo
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import animation  

E = 0.05 # Electric field
B = 1.0 # Magnetic field
exit_velocity = 100.0 # Velocity at which the particle exits the fields
exit_radius = 100.0 # Radius at which the particle exits the fields

def f_polar(r, t, q, m):
    radius, theta, z, z_velocity = r
    fr =  E * q / m 
    ftheta = -q * B / m
    fz = z_velocity
    
    # if statement (limit)
    velocity = np.sqrt(z_velocity**2 + (radius * ftheta)**2)
    if velocity >= exit_velocity or radius >= exit_radius:
        fr = 0.0
        ftheta = 0.0
        
    fzv = E * t
    return fr, ftheta, fz, fzv

def cyclotron(q, m):
    r = [1.0, 0.0, 0.0, 0.0] 
    N = int(1E3)
    tpoints = np.linspace(0, 50, N)
    rpoints = odeint(f_polar, r, tpoints, args=(q, m))
    
    radius_points = rpoints[:, 0]
    theta_points = rpoints[:, 1]

    x_points = radius_points * np.cos(theta_points)
    y_points = radius_points * np.sin(theta_points)
    z_points = 0.5 * E * tpoints**2

    return x_points, y_points, z_points, tpoints, radius_points

# particle 1st curve
charge1 = 1
mass1 = 1.0

# particle 2nd curve
charge2 = 1.4
mass2 = 1.1

# particle 3rd curve
charge3 = 0.8
mass3 = 1.1

x_points1, y_points1, z_points1, tpoints1,r_points1 = cyclotron(charge1, mass1)
x_points2, y_points2, z_points2, tpoints2,r_points2 = cyclotron(charge2, mass2)
x_points3, y_points3, z_points3, tpoints3,r_points3 = cyclotron(charge3, mass3)
#%%
# kinetic energy for 1st particle
ke1=np.array((0.5*mass1*((2*np.pi*r_points1)/tpoints1)))
print(' Initial energy for first particle:', ke1[-1])
print(' Final energy for first particle:', ke1[99])

# kinetic energy for 2nd particle
ke2=np.array(0.5*mass2*((2*np.pi*r_points2)/tpoints2))
print(' Initial energy for first particle:', ke2[-1])
print(' Final energy for first particle:', ke2[99])

# kinetic energy for 3rd particle
ke3=np.array(0.5*mass3*((2*np.pi*r_points3)/tpoints3))
print(' Initial energy for first particle:', ke3[-1])
print(' Final energy for first particle:', ke3[99])

plt.figure()
plt.plot(x_points1, y_points1, label='Trajectory 1st particle', c='m')  
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#%%
# Trajectory 1
plt.figure()
plt.suptitle('Particle Motion in Electric and Magnetic Fields')  

plt.subplot(131)
plt.plot(x_points1, y_points1, label='Trajectory 1st particle', c='m')  
plt.ylabel('y')

# Trajectory 2
plt.subplot(132)
plt.plot(x_points2, y_points2, label='Trajectory 2nd particle') 
plt.xlabel('x')
plt.ylabel('y')

# Trajectory 3
plt.subplot(133)
plt.plot(x_points3, y_points3, label='Trajectory 3rd particle', c='b') 
plt.xlabel('x')
plt.ylabel('y')
 
plt.show()
#%%

plt.figure()
plt.suptitle('Radial Position over Time')

# r_points against tpoints 1
plt.subplot(131)
plt.plot(tpoints1, x_points1,c='m')
plt.xlabel('Time')
plt.ylabel('Radial Position (r)')

# r_points against tpoints 2
plt.subplot(132)
plt.plot(tpoints2, x_points2)
plt.xlabel('Time')
plt.ylabel('Radial Position (r)')


# r_points against tpoints 3
plt.subplot(133)
plt.plot(tpoints3, x_points3,c='b')
plt.xlabel('Time')
plt.ylabel('Radial Position (r)')

plt.tight_layout()
plt.legend()
plt.show()

#%%
# Position over time
plt.figure()
plt.plot(tpoints1, np.sqrt(x_points1**2 + y_points1**2), label='Particle 1', color='purple', linestyle='-')
plt.plot(tpoints2, np.sqrt(x_points2**2 + y_points2**2), label='Particle 2', color='blue', linestyle='--')
plt.plot(tpoints3, np.sqrt(x_points3**2 + y_points3**2), label='Particle 3', color='green', linestyle=':')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Particle Motion in Electric and Magnetic Fields')
plt.legend()
plt.show()

#%%
#Animation  1 
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot([], [], lw=1)

def init():
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return line,

def update(frame):
    line.set_data(x_points1[:frame], y_points1[:frame])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(x_points1), init_func=init, blit=True, interval=20)
plt.show()
#%%
# animation 2

fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot([], [], lw=1)

def init():
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return line,

def update(frame):
    line.set_data(x_points2[:frame], y_points2[:frame])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(x_points2), init_func=init, blit=True, interval=20)
plt.show()

#%%
# animation 3
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot([], [], lw=1)

def init():
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return line,

def update(frame):
    line.set_data(x_points3[:frame], y_points3[:frame])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(x_points3), init_func=init, blit=True, interval=20)
plt.show()


#%%

rho = 50
alpha = 2.633E-3
p = 1.735

dx=1E-3                 
Estart=[ke1[-1],ke2[-1],ke3[-1]]

plot_xvals = []
plot_yvals = []
for E in Estart:
    x=0 
    x_vals = []
    dEdx_vals = []
    while 0 < E > 0.2: 
        x=x+dx  
        dEdx=E**(1-p)/(rho*alpha*p) 
        dE=dEdx*dx 
        E=E-dE
        if E < 0: break
        x_vals.append(x)
        dEdx_vals.append(dEdx)
    while x < E:
        x = x + dx
        x_vals.append(x)
        dEdx_vals.append(0)
    
    plot_xvals.append(x_vals)
    plot_yvals.append(dEdx_vals)

plt.figure()
plt.plot(plot_xvals[0],plot_yvals[0],'m--',lw=1, label='Particle 1')
plt.plot(plot_xvals[1],plot_yvals[1],'b--',lw=1, label='Particle 2')
plt.plot(plot_xvals[2],plot_yvals[2],'g--',lw=1, label='Particle 3')
plt.xlabel('Depth (cm)',fontsize=15)
plt.legend(loc='best')
plt.ylabel(r'$\mathrm{d}E/\mathrm{d}x\;(\mathrm{MeV}\;\mathrm{cm}^2\;\mathrm{g}^{-1})$',fontsize=15)
plt.show()

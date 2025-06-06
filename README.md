# Cyclotron Simulation in Python – DSC 250

This project simulates the motion of a charged particle in a cyclotron using Python. A cyclotron accelerates particles using a constant magnetic field and an alternating electric field, causing the particle to spiral outward as it gains energy.

---

##  Overview

- Simulates a classical cyclotron setup in 2D
- Uses Lorentz force calculations to update particle motion
- Visualizes particle trajectory and kinetic energy growth over time
- Demonstrates key physics principles: uniform circular motion, resonance, and relativistic effects (if extended)

---

## Physical Model

The simulation incorporates:
- A perpendicular magnetic field (**B**) that causes circular motion
- An oscillating electric field (**E**) applied between "dees" to accelerate the particle
- Numerical integration to update particle velocity and position over time
- Conservation of energy and basic relativistic limits (optional)

---

## Features

- Adjustable particle mass, charge, and initial conditions
- Variable magnetic field strength and frequency
- Plots of particle trajectory and radius vs. time
- Optional energy or phase plots

---

## Libraries

- `numpy` – numerical arrays and vector math
- `matplotlib` – visualization of particle motion
- `scipy` (optional) – advanced integration or Fourier analysis

""" Solve pressure equation. """

import numpy as np
import ressim

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt

np.random.seed(42)  # for reproducibility

grid = ressim.Grid(nx=64, ny=64, lx=1.0, ly=1.0)  # unit square, 64x64 grid
k = np.exp(np.load('perm.npy').reshape(grid.shape))  # load log-permeability, convert to absolute with exp()
q = np.zeros(grid.shape); q[0,0]=1; q[-1,-1]=-1  # source term: corner-to-corner flow (a.k.a. quarter-five spot)

# instantiate solver
solver = ressim.PressureEquation(grid=grid, q=q, k=k)
# solve
solver.step()
p = solver.p

# visualize
plt.figure()
plt.imshow(p)
plt.colorbar()
plt.savefig('pressure.png')

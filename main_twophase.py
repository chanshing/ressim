""" Transient two-phase (oil-water) flow """

from time import time

import numpy as np
import functools

import ressim
import utils

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['image.cmap'] = 'jet'
import matplotlib.pyplot as plt

np.random.seed(42)  # for reproducibility

grid = ressim.Grid(nx=64, ny=64, lx=1.0, ly=1.0)  # unit square, 64x64 grid
k = np.exp(np.load('perm.npy').reshape(grid.shape))  # load log-permeability, convert to absolute with exp()
q = np.zeros(grid.shape); q[0,0]=1; q[-1,-1]=-1  # source term: corner-to-corner flow (a.k.a. quarter-five spot)

mu_w, mu_o = 1.0, 10.  # viscosities
s_wir, s_oir = 0.2, 0.2  # irreducible saturations

phi = np.ones(grid.shape)*0.2  # uniform porosity
s0 = np.ones(grid.shape) * s_wir  # initial water saturation equals s_wir
dt = 1e-3  # timestep

mobi_fn = functools.partial(utils.quadratic_mobility, mu_w=mu_w, mu_o=mu_o, s_wir=s_wir, s_oir=s_oir)  # quadratic mobility model
lamb_fn = functools.partial(utils.lamb_fn, mobi_fn=mobi_fn)  # total mobility function
f_fn = functools.partial(utils.f_fn, mobi_fn=mobi_fn)  # water fractional flow function

# (Optional) derivative of water fractional flow
# This is to compute the jacobian of the residual to accelerate the
# saturation solver. If not provided, the jacobian is approximated in the
# solver.
df_fn = functools.partial(utils.df_fn, mobi_fn=mobi_fn)

# instantiate solvers
solverP = ressim.PressureEquation(grid, q=q, k=k, lamb_fn=lamb_fn)
solverS = ressim.SaturationEquation(grid, q=q, phi=phi, s=s0, f_fn=f_fn, df_fn=df_fn)

# solve for 25 timesteps
nstep = 25
s_list = []
for i in range(nstep):
    before = time()

    # solve pressure
    solverP.s = solverS.s
    solverP.step()

    # solve saturation
    solverS.v = solverP.v
    solverS.step(dt)

    after = time()
    print '[{}/{}]: this loop took {} secs'.format(i+1, nstep, after - before)

    s_list.append(solverS.s)

# visualize
fig, axs = plt.subplots(5,5, figsize=(8,8))
fig.subplots_adjust(wspace=.1, hspace=.1, left=0, right=1, bottom=0, top=1)
for ax, s in zip(axs.ravel(), s_list):
    ax.imshow(s)
    ax.axis('off')
fig.savefig('saturations.png', bbox_inches=0, pad_inches=0)

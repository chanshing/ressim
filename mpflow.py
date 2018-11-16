import numpy
from numpy import array, asarray, copy, zeros, ones, maximum, minimum
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

import warnings

import utils

class Grid(object):
    """
    Simple rectangular grid.

    Parameters
    ----------
    nx, ny : int, int
        Grid resolution
    lx, ly : float, float, optional
        Grid physical dimensions. Defaults to lx=1.0, ly=1.0 (unit square)

    Attributes
    ----------
    vol : float
        cell volume
    dx, dy : float, float
        cell dimensions
    ncell : int
        number of cells
    """
    def __init__(self, nx, ny, lx=1.0, ly=1.0):
        self.nx, self.ny = nx, ny
        self.lx, self.ly = float(lx), float(ly)

        self.ncell = nx*ny
        self.dx, self.dy = self.lx/nx, self.ly/ny
        self.vol = self.dx*self.dy

class PressureSolver(object):
    """
    Solver for the pressure equation.

    Parameters
    ----------
    grid :
        Grid object defining the domain

    k : ndarray, shape (ny, nx)
        Permeability

    q : ndarray, shape (ny, nx) | (ny*nx,)
        Integrated source term.

    diri : list of (int, float) tuples, optional
        Dirichlet boundary conditions, e.g. [(i1, val1), (i2, val2), ...]
        means pressure values val1 at cell i1, val2 at cell i2, etc. Defaults
        to [(ny*nx/2, 0.0)], i.e. zero pressure at center of the grid.

    mobi : ndarray, shape (ny, nx) | (ny*nx,) OR float, optional
        Total mobility. A single float means mobility is uniform in the
        domain. Defaults to 1.0

    Attributes
    ----------
    p : ndarray, shape (ny, nx)
        Pressure

    v : dict of ndarray
        'x' : ndarray, shape (ny, nx+1)
            Flux in x-direction
        'y' : ndarray, shape (ny+1, nx)
            Flux in y-direction

    Methods
    -------
    step() :
        Main method that solves the pressure equation to obtain pressure and
        flux, stored at self.p and self.v
    """
    def __init__(self, grid, k, q, mobi=1.0, diri=None):
        self.grid, self.k, self.q = grid, k, q
        self.diri = diri
        self.mobi = mobi

        self.p, self.v = None, None

    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, k):
        assert numpy.all(k > 0), "Invalid negative permeability. Perhaps forgot to exp(k)?"
        self.__k = k

    @property
    def diri(self):
        return self.__diri

    @diri.setter
    def diri(self, diri):
        """ default is zero at center of the grid """
        if diri is None:
            n = self.grid.ncell
            self.__diri = [(int(n/2), 0.0)]

    def step(self):
        grid, k = self.grid, self.k
        mobi = self.mobi

        nx, ny = grid.nx, grid.ny

        if isinstance(mobi, numpy.ndarray):
            mobi = mobi.reshape(*k.shape)
        k = k * mobi

        mat, tx, ty = transmi(grid, k)
        q = copy(self.q).ravel()
        impose_diri(mat, q, self.diri)  # inplace op on mat, q

        # pressure
        p = spsolve(mat, q)
        p = p.reshape(ny, nx)
        # flux
        v = {'x':zeros((ny,nx+1)), 'y':zeros((ny+1,nx))}
        v['x'][:,1:nx] = (p[:,0:nx-1]-p[:,1:nx])*tx[:,1:nx]
        v['y'][1:ny,:] = (p[0:ny-1,:]-p[1:ny,:])*ty[1:ny,:]

        self.p, self.v = p, v

# def SaturationSolver(object):
#     def __init__(self, grid, v, q, s_init=None, frac_fn=None, **frac_kws):

def transmi(grid, k):
    """ construct transmisibility matrix """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    n = grid.ncell

    k = k.reshape(ny, nx)
    kinv = 1.0/k

    ax = 2*dy/dx; tx = zeros((ny,nx+1))
    ay = 2*dx/dy; ty = zeros((ny+1,nx))

    tx[:,1:nx] = ax/(kinv[:,0:nx-1]+kinv[:,1:nx])
    ty[1:ny,:] = ay/(kinv[0:ny-1,:]+kinv[1:ny,:])

    x1 = tx[:,0:nx].reshape(n); x2 = tx[:,1:nx+1].reshape(n)
    y1 = ty[0:ny,:].reshape(n); y2 = ty[1:ny+1,:].reshape(n)

    data = [-y2, -x2, x1+x2+y1+y2, -x1, -y1]
    diags = [-nx, -1, 0, 1, nx]
    mat = spdiags(data, diags, n, n, format='csr')

    return mat, tx, ty

def convecti(grid, v):
    """ construct convection matrix with upwind scheme """
    nx, ny = grid.nx, grid.ny
    n = grid.ncell

    xn = minimum(v['x'], 0); x1 = xn[:,:,0:nx].reshape(n)
    yn = minimum(v['y'], 0); y1 = yn[:,0:ny,:].reshape(n)
    xp = maximum(v['x'], 0); x2 = xp[:,:,1:nx+1].reshape(n)
    yp = maximum(v['y'], 0); y2 = yp[:,1:ny+1,:].reshape(n)

    data = [-y2, -x2, x2-x1+y2-y1, x1, y1]
    diags = [-nx, -1, 0, 1, nx]
    mat = spdiags(data, diags, n, n, format='csr')

    return mat

def impose_diri(mat, q, diri):
    """
    Impose Dirichlet boundary conditions. NOTE: inplace operation on mat, q
    For example, to impose a pressure value 99 at the first cell:

    mat = [[  1   0  ...  0  ]
           [ a21 a22 ... a2n ]
           ...
           [ an1 an2 ... ann ]]

    q = [99 q2 ... qn]
    """
    for i, val in diri:
        utils.csr_row_set_nz_to_val(mat, i, 0.0)
        mat[i,i] = 1.0
        q[i] = val
    mat.eliminate_zeros()

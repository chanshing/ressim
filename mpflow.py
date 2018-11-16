import numpy
from numpy import array, asarray, copy, zeros, ones, maximum, minimum
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy import optimize

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
        self.lx, self.ly = lx, ly

        self.update()

    def update(self):
        if all(hasattr(self, key) for key in ['nx', 'ny', 'lx', 'ly']):
            self.shape = (self.ny, self.nx)
            self.ncell = self.nx*self.ny
            self.dx, self.dy = self.lx/self.nx, self.ly/self.ny
            self.vol = self.dx*self.dy

    @property
    def nx(self):
        return self.__nx

    @property
    def ny(self):
        return self.__ny

    @property
    def lx(self):
        return self.__lx

    @property
    def ly(self):
        return self.__ly

    @nx.setter
    def nx(self, nx):
        self.__nx = int(nx)
        self.update()

    @ny.setter
    def ny(self, ny):
        self.__ny = int(ny)
        self.update()

    @lx.setter
    def lx(self, lx):
        self.__lx = float(lx)
        self.update()

    @ly.setter
    def ly(self, ly):
        self.__ly = float(ly)
        self.update()

class BaseSolver(object):

    @property
    def k(self):
        return self.__k

    @property
    def lamb(self):
        return self.__lamb

    @property
    def q(self):
        return self.__q

    @property
    def s(self):
        return self.__s

    @property
    def phi(self):
        return self.__phi

    @k.setter
    def k(self, k):
        assert numpy.all(k > 0), "Non-positive permeability. Perhaps forgot to exp(k)?"
        self.__k = k

    @lamb.setter
    def lamb(self, lamb):
        assert numpy.all(lamb > 0), "Non-positive mobility"
        self.__lamb = lamb

    @q.setter
    def q(self, q):
        assert numpy.sum(q) == 0, "Unbalanced source term"
        self.__q = q

    @s.setter
    def s(self, s):
        assert numpy.all(s >= 0) and numpy.all(s <= 1), "Saturation not in [0,1]"
        self.__s = s

    @phi.setter
    def phi(self, phi):
        assert numpy.all(phi >= 0) and numpy.all(phi <= 1), "Porosity not in [0,1]"
        self.__phi = phi

class PressureSolver(BaseSolver):
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

    lamb : ndarray, shape (ny, nx) | (ny*nx,) OR float, optional
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
    def __init__(self, grid, k, q, lamb=1.0, diri=None):
        self.grid, self.k, self.q = grid, k, q
        self.lamb = lamb

        if diri is None:
            # defaults to zero at center of the grid
            self.diri = [(int(self.grid.ncell/2), 0.0)]

    def step(self):
        grid, k = self.grid, self.k
        lamb = self.lamb

        nx, ny = grid.nx, grid.ny

        if isinstance(lamb, numpy.ndarray):
            lamb = lamb.reshape(*k.shape)
        k = k * lamb

        mat, tx, ty = transmi(grid, k)
        q = copy(self.q).reshape(grid.ncell)
        impose_diri(mat, q, self.diri)  # inplace op on mat, q

        # pressure
        p = spsolve(mat, q)
        p = p.reshape(*grid.shape)
        # flux
        v = {'x':zeros((ny,nx+1)), 'y':zeros((ny+1,nx))}
        v['x'][:,1:nx] = (p[:,0:nx-1]-p[:,1:nx])*tx[:,1:nx]
        v['y'][1:ny,:] = (p[0:ny-1,:]-p[1:ny,:])*ty[1:ny,:]

        # update self.p, self.v
        self.p, self.v = p, v

class SaturationSolver(BaseSolver):
    def __init__(self, grid, q, phi, s, f_fn, v=None):
        self.grid, self.q, self.phi, self.s, self.f_fn = grid, q, phi, s, f_fn
        self.v = v

    @staticmethod
    def qw(q, frac):
        # water injection, mixed production
        qw = maximum(q,0) + frac*minimum(q,0)
        return qw

    def step(self, dt):
        grid, q, phi, s = self.grid, self.q, self.phi, self.s
        v = self.v
        f_fn = self.f_fn

        alpha = float(dt) / (grid.vol * phi)
        mat = convecti(grid, v)

        s = s.reshape(grid.ncell)
        q = q.reshape(grid.ncell)
        if isinstance(alpha, numpy.ndarray):
            alpha = alpha.reshape(grid.ncenll)

        def residual(s1):
            frac = f_fn(s1)
            qw = self.qw(q, frac)
            return s1 - s + alpha * (mat.dot(frac) - qw)
        sol = optimize.root(residual, x0=s, method='krylov')

        # clip to ensure solution in [0,1]; update self.s
        self.s = numpy.clip(sol.x.reshape(*grid.shape), 0., 1.)

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

    xn = minimum(v['x'], 0); x1 = xn[:,0:nx].reshape(n)
    yn = minimum(v['y'], 0); y1 = yn[0:ny,:].reshape(n)
    xp = maximum(v['x'], 0); x2 = xp[:,1:nx+1].reshape(n)
    yp = maximum(v['y'], 0); y2 = yp[1:ny+1,:].reshape(n)

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

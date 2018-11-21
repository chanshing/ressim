""" A Module for reservoir simulation in Python """

import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg
import scipy.optimize

class Grid(object):
    """
    Simple rectangular grid.

    Attributes
    ----------
    nx, ny : int, int
        Grid resolution

    lx, ly : float, float, optional
        Grid physical dimensions. (default lx=1.0, ly=1.0, i.e. unit square)

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

    @property
    def shape(self):
        return (self.ny, self.nx)

    @property
    def ncell(self):
        return self.nx*self.ny

    @property
    def vol(self):
        return self.dx*self.dy

    @property
    def dx(self):
        return self.lx/self.nx

    @property
    def dy(self):
        return self.ly/self.ny

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

    @ny.setter
    def ny(self, ny):
        self.__ny = int(ny)

    @lx.setter
    def lx(self, lx):
        self.__lx = float(lx)

    @ly.setter
    def ly(self, ly):
        self.__ly = float(ly)

class Parameters(object):
    """ Container for equation paremeters with minimal checks """

    @property
    def grid(self):
        return self.__grid

    @property
    def k(self):
        return self.__k

    @property
    def q(self):
        return self.__q

    @property
    def s(self):
        return self.__s

    @property
    def phi(self):
        return self.__phi

    @property
    def v(self):
        return self.__v

    @property
    def lamb_fn(self):
        return self.__lamb_fn

    @property
    def f_fn(self):
        return self.__f_fn

    @grid.setter
    def grid(self, grid):
        if grid is not None:
            assert isinstance(grid, Grid)
            self.__grid = grid

    @k.setter
    def k(self, k):
        if k is not None:
            assert isinstance(k, np.ndarray)
            assert np.all(k > 0), "Non-positive permeability. Perhaps forgot to exp(k)?"
            self.__k = k

    @q.setter
    def q(self, q):
        if q is not None:
            assert isinstance(q, np.ndarray)
            assert np.sum(q) == 0, "Unbalanced source term"
            self.__q = q

    @s.setter
    def s(self, s):
        if s is not None:
            assert isinstance(s, np.ndarray)
            assert np.all(s >= 0) and np.all(s <= 1), "Saturation not in [0,1]"
            self.__s = s

    @phi.setter
    def phi(self, phi):
        if phi is not None:
            assert isinstance(phi, np.ndarray)
            assert np.all(phi >= 0) and np.all(phi <= 1), "Porosity not in [0,1]"
            self.__phi = phi

    @v.setter
    def v(self, v):
        if v is not None:
            assert isinstance(v, dict)
            assert isinstance(v['x'], np.ndarray)
            assert isinstance(v['y'], np.ndarray)
            self.__v = v

    @lamb_fn.setter
    def lamb_fn(self, lamb_fn):
        if lamb_fn is not None:
            assert callable(lamb_fn)
            self.__lamb_fn = lamb_fn

    @f_fn.setter
    def f_fn(self, f_fn):
        if f_fn is not None:
            assert callable(f_fn)
            self.__f_fn = f_fn

class PressureEquation(Parameters):
    """
    Pressure equation

    Attributes
    ----------
    grid :
        Grid object defining the domain

    q : ndarray, shape (ny, nx) | (ny*nx,)
        Integrated source term.

    k : ndarray, shape (ny, nx)
        Permeability

    diri : list of (int, float) tuples
        Dirichlet boundary conditions, e.g. [(i1, val1), (i2, val2), ...]
        means pressure values val1 at cell i1, val2 at cell i2, etc. Defaults
        to [(ny*nx/2, 0.0)], i.e. zero pressure at center of the grid.

    lamb_fn : callable
        Total mobility function lamb_fn(s)

    s : ndarray, shape (ny, nx) | (ny*nx,)
        Saturation

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
        Solve the pressure equation to obtain pressure and flux. Update
        self.p and self.v

    solve(mat, q):
        Method to solve the system of linear equations. Default is
        scipy.sparse.linalg.spsolve(mat, q)
        You can override this method to use a different solver.
    """
    def __init__(self, grid=None, q=None, k=None, diri=None, lamb_fn=None, s=None):
        self.grid, self.q, self.k = grid, q, k
        self.diri = diri
        self.lamb_fn = lamb_fn
        self.s = s

    @property
    def diri(self):
        """ Default to zero at center of the grid """
        if self.__diri is None:
            return [(int(self.grid.ncell/2), 0.0)]
        return self.__diri

    @diri.setter
    def diri(self, diri):
        self.__diri = diri

    def step(self):
        grid, q, k = self.grid, self.q, self.k
        diri = self.diri

        if hasattr(self, 'lamb_fn'):
            k = k * self.lamb_fn(self.s).reshape(*grid.shape)

        mat, tx, ty = transmi(grid, k)
        q = np.copy(q).reshape(grid.ncell)
        impose_diri(mat, q, diri)  # inplace op on mat, q

        # pressure
        p = self.solve(mat, q)
        p = p.reshape(*grid.shape)
        # flux
        nx, ny = grid.nx, grid.ny
        v = {'x':np.zeros((ny,nx+1)), 'y':np.zeros((ny+1,nx))}
        v['x'][:,1:nx] = (p[:,0:nx-1]-p[:,1:nx])*tx[:,1:nx]
        v['y'][1:ny,:] = (p[0:ny-1,:]-p[1:ny,:])*ty[1:ny,:]

        self.p, self.v = p, v

    def solve(self, mat, q, **kws):
        return scipy.sparse.linalg.spsolve(mat, q, **kws)

class SaturationEquation(Parameters):
    """
    Saturation equation

    Attributes
    ----------
    grid :
        Grid object defining the domain

    q : ndarray, shape (ny, nx) | (ny*nx,)
        Integrated source term.

    phi : ndarray, shape (ny, nx) | (ny*nx,)
        Porosity

    f_fn : callable
        Water fractional flow function f_fn(s)

    v : dict of ndarray
        'x' : ndarray, shape (ny, nx+1)
            Flux in x-direction
        'y' : ndarray, shape (ny+1, nx)
            Flux in y-direction

    df_fn : callable (optional)
        Derivative (element-wise) of water fractional flow function df_fn(s).
        It is used to compute the jacobian of the residual function. If None,
        the jacobian is approximated by the solver (which can be slow).

    s : ndarray, shape (ny, nx) | (ny*nx,)
        Saturation

    Methods
    -------
    step(dt) :
        Solve saturation forward in time by dt. Update self.s

    solve(residual, s0, residual_jac=None, **kws) :
        Method to perform the minimization of the residual. Default is
        scipy.optimize.least_squares(residual, x0=s0, jac=residual_jac, method='trf', tr_solver='lsmr')
        You can override this method to use a different solver.
    """
    def __init__(self, grid=None, q=None, phi=None, s=None, f_fn=None, v=None, df_fn=None):
        self.grid, self.q, self.phi, self.s, self.f_fn = grid, q, phi, s, f_fn
        self.v = v
        self.df_fn = df_fn

    @property
    def df_fn(self):
        return self.__df_fn

    @df_fn.setter
    def df_fn(self, df_fn):
        if df_fn is not None:
            assert(callable(df_fn))
            self.__df_fn = df_fn

    def step(self, dt):
        grid, q, phi, s = self.grid, self.q, self.phi, self.s
        v = self.v
        f_fn = self.f_fn

        alpha = float(dt) / (grid.vol * phi)
        mat = convecti(grid, v)

        s = s.reshape(grid.ncell)
        q = q.reshape(grid.ncell)
        alpha = alpha.reshape(grid.ncell)

        def residual(s1):
            f = f_fn(s1)
            qp = np.maximum(q,0)
            qn = np.minimum(q,0)
            r = s1 - s + alpha * (mat.dot(f) - (qp + f*qn))
            return r

        residual_jac = None
        if hasattr(self, 'df_fn'):
            def residual_jac(s1):
                df = self.df_fn(s1)
                qn = np.minimum(q,0)
                eye = spa.eye(len(s1))
                df_eye = spa.diags(df, 0, shape=(len(s1), len(s1)))
                alpha_eye = spa.diags(alpha, 0, shape=(len(s1), len(s1)))
                qn_eye = spa.diags(qn, 0, shape=(len(s1), len(s1)))
                dr = eye + (alpha_eye.dot(mat - qn_eye)).dot(df_eye)
                return dr

        sol = self.solve(residual, s0=s, residual_jac=residual_jac)
        self.s = np.clip(sol.x.reshape(*grid.shape), 0., 1.)  # clip to ensure within [0,1]

    def solve(self, residual, s0, residual_jac=None):
        if residual_jac is None:
            residual_jac = '2-point'
        return scipy.optimize.least_squares(residual, x0=s0, jac=residual_jac, method='trf', tr_solver='lsmr')

def transmi(grid, k):
    """ Construct transmisibility matrix with two point flux approximation """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy
    n = grid.ncell

    k = k.reshape(*grid.shape)
    kinv = 1.0/k

    ax = 2*dy/dx; tx = np.zeros((ny,nx+1))
    ay = 2*dx/dy; ty = np.zeros((ny+1,nx))

    tx[:,1:nx] = ax/(kinv[:,0:nx-1]+kinv[:,1:nx])
    ty[1:ny,:] = ay/(kinv[0:ny-1,:]+kinv[1:ny,:])

    x1 = tx[:,0:nx].reshape(n); x2 = tx[:,1:nx+1].reshape(n)
    y1 = ty[0:ny,:].reshape(n); y2 = ty[1:ny+1,:].reshape(n)

    data = [-y2, -x2, x1+x2+y1+y2, -x1, -y1]
    diags = [-nx, -1, 0, 1, nx]
    mat = spa.spdiags(data, diags, n, n, format='csr')

    return mat, tx, ty

def convecti(grid, v):
    """ Construct convection matrix with upwind scheme """
    nx, ny = grid.nx, grid.ny
    n = grid.ncell

    xn = np.minimum(v['x'], 0); x1 = xn[:,0:nx].reshape(n)
    yn = np.minimum(v['y'], 0); y1 = yn[0:ny,:].reshape(n)
    xp = np.maximum(v['x'], 0); x2 = xp[:,1:nx+1].reshape(n)
    yp = np.maximum(v['y'], 0); y2 = yp[1:ny+1,:].reshape(n)

    data = [-y2, -x2, x2-x1+y2-y1, x1, y1]
    diags = [-nx, -1, 0, 1, nx]
    mat = spa.spdiags(data, diags, n, n, format='csr')

    return mat

def impose_diri(mat, q, diri):
    """ Impose Dirichlet boundary conditions. NOTE: inplace operation on mat, q
    For example, to impose a pressure value 99 at the first cell:

    mat = [[  1   0  ...  0  ]
           [ a21 a22 ... a2n ]
           ...
           [ an1 an2 ... ann ]]

    q = [99 q2 ... qn]
    """
    for i, val in diri:
        csr_row_set_nz_to_val(mat, i, 0.0)
        mat[i,i] = 1.0
        q[i] = val
    mat.eliminate_zeros()

def csr_row_set_nz_to_val(csr, row, value=0):
    """ Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly. """
    if not isinstance(csr, spa.csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row+1]] = value

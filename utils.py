import numpy
import scipy

def csr_row_set_nz_to_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, scipy.sparse.csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row+1]] = value

class BaseMobility(object):
    def __init__(self, vw=1.0, vo=1.0, swc=0.0, sor=0.0):
        self.vw, self.vo, self.swc, self.sor = vw, vo, swc, sor

    @property
    def vw(self):
        return self.__vw

    @property
    def vo(self):
        return self.__vo

    @property
    def swc(self):
        return self.__swc

    @property
    def sor(self):
        return self.__sor

    @vw.setter
    def vw(self, vw):
        self.__vw = float(vw)

    @vo.setter
    def vo(self, vo):
        self.__vo = float(vo)

    @swc.setter
    def swc(self, swc):
        self.__swc = float(swc)

    @sor.setter
    def sor(self, sor):
        self.__sor = float(sor)

class LinearMobility(BaseMobility):
    def __call__(self, s):
        _s = (s-self.swc)/(1.0-self.swc-self.sor)
        lamb_w = _s/self.vw
        lamb_o = (1.0-_s)/self.vo
        # clip to ensure within [0,1]
        lamb_w, lamb_o = numpy.clip(lamb_w, 0., 1.), numpy.clip(lamb_o, 0., 1.)
        return lamb_w, lamb_o

class QuadraticMobility(BaseMobility):
    def __call__(self, s):
        _s = (s-self.swc)/(1.0-self.swc-self.sor)
        lamb_w = _s**2/self.vw
        lamb_o = (1.0-_s)**2/self.vo
        # clip to ensure within [0,1]
        lamb_w, lamb_o = numpy.clip(lamb_w, 0., 1.), numpy.clip(lamb_o, 0., 1.)
        return lamb_w, lamb_o

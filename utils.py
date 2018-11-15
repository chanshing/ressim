import scipy

def csr_row_set_nz_to_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, scipy.sparse.csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row+1]] = value

class MobilityFunction(object):
    def __init__(self, vw=1.0, vo=1.0, swc=0.0, sor=0.0, model='linear'):
        self.vw, self.vo, self.swc, self.sor = float(vw), float(vo), float(swc), float(sor)
        self.model = model

    def __call__(self, s):
        if self.model == 'linear':
            _s = (s-self.swc)/(1.0-self.swc-self.sor)
            mw = _s/self.vw
            mo = 1.0-_s/self.vo
        elif self.model == 'quadratic':
            _s = (s-self.swc)/(1.0-self.swc-self.sor)
            mw = _s**2/self.vw
            mo = (1.0-_s)**2/self.vo
        else: raise ValueError('choose linear or quadratic')

        return mw, mo

import numpy as np
import healpy as hp

def cli(cl):
    """
    Pseudo-inverse of the input cl array.
    Parameters:
    -----------
    cl : np.ndarray
        The input cl array.
    Returns:
    --------
    np.ndarray
        The pseudo-inverse of the input cl array.
    """
    ret = np.zeros_like(cl)
    ii = np.where(cl != 0)
    ret[ii] = 1. / cl[ii]
    return ret

def slice_alms(teb, lmax_new):
    """
    Slice the alm array.
    Parameters:
    -----------
    teb : np.ndarray
        The input alm array.
    lmax_new : int
        Maximum l.
    Returns:
    --------
    np.ndarray
        The sliced alm array.
    """
    lmax = hp.Alm.getlmax(len(teb[0]))
    if lmax_new > lmax:
        raise ValueError('lmax_new must be smaller or equal to lmax')
    elif lmax_new == lmax:
        return teb
    else:
        teb_new = np.zeros((len(teb), hp.Alm.getsize(lmax_new)), dtype=teb.dtype)
        indices_full = hp.Alm.getidx(lmax,*hp.Alm.getlm(lmax_new))
        indices_new = hp.Alm.getidx(lmax_new,*hp.Alm.getlm(lmax_new))
        teb_new[:,indices_new] = teb[:,indices_full]
        return teb_new
import numpy as np
import numba

def ccdelta(f, t, tout, left, right):
    """
    Compute the cross correlation of a given function with a mixture of dirac
    deltas, i.e. evaluate g(t) = 1/N sum_i f(t_i - t).
    
    Parameters
    ----------
    f : function
        A numba nopython jitted function with signature scalar->scalar.
    t : array (..., N)
        The locations of deltas. Must be sorted along the last axis.
    tout : array (..., K)
        The points where the cross correlation is evaluated. Must be sorted
        along the last axis.
    left, right : scalar
        The support of f. f is assumed to be zero outside of it and not
        evaluated.
    
    Return
    ------
    out : array (..., K)
        The cross correlation evaluated on tout. The shape is determined by
        broadcasting t with tout along all axes but the last.
    """
    assert callable(f)
    t = np.asarray(t)
    assert len(t.shape) >= 1
    tout = np.asarray(tout)
    assert len(tout.shape) >= 1
    assert np.isscalar(left)
    assert np.isscalar(right)
    
    shape = np.broadcast(t[..., 0], tout[..., 0]).shape
    t = np.broadcast_to(t, shape + t.shape[-1:]).reshape(-1, t.shape[-1])
    tout = np.broadcast_to(tout, shape + tout.shape[-1:]).reshape(-1, tout.shape[-1])
    out = np.zeros(shape + tout.shape[-1:])
    
    _ccdelta(f, t, tout, left, right, out.reshape(-1, out.shape[-1]))
    
    return out

@numba.njit
def _ccdelta(f, t, tout, left, right, out):
    """
    Compiled implementation of ccdelta. The out array must be initialized to
    zero. The shapes of t, tout, out must be (M, N), (M, K), (M, K).
    """
    for iouter in numba.prange(len(out)):
        td = t[iouter]
        tg = tout[iouter]
        g = out[iouter]
        
        igmin = 0
        for ti in td:
            igmin += np.searchsorted(tg[igmin:], ti - right)
            if igmin >= len(tg):
                break
            ig = igmin
            while ig < len(tg) and tg[ig] <= ti - left:
                g[ig] += f(ti - tg[ig])
                ig += 1
        
        g /= len(td)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from scipy import interpolate
    
    fig = plt.figure('ccdelta')
    fig.clf()
    
    ax = fig.subplots(1, 1)
    
    @numba.njit
    def f(t):
        return 1 if 0 < t < 1 else 0
    
    gen = np.random.default_rng(202012201928)
    t = np.sort(gen.uniform(0, 5, 25))
    tout = np.linspace(-1, 6, 10000)
    
    out = ccdelta(f, t, tout, 0, 1)
    out2 = ccdelta(f, t[None, None], tout[None], 0, 1)[0, 0]
    assert np.array_equal(out, out2)
    out *= len(t)
    
    y = interpolate.interp1d(tout, out)
    ax.plot(tout, out, color='lightgray')
    ax.plot(t, y(t), 'x', color='black')
    ax.plot(t - 1, y(t - 1), '+', color='black')
        
    fig.tight_layout()
    plt.show()

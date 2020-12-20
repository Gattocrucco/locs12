import numpy as np
import numba

@numba.njit
def ccdelta(f, t, tout, left, right):
    """
    Compute the cross correlation of a given function with a mixture of dirac
    deltas, i.e. evaluate g(t) = 1/N sum_i f(t_i - t).
    
    Parameters
    ----------
    f : function
        A nopython-mode numba jitted function with signature scalar->scalar.
    t : array (M, N)
        The locations of deltas. Must be sorted along the last axis. The
        computation is broadcasted on the first axis.
    tout : array (M, K)
        The points where the cross correlation is evaluated. Must be sorted
        along the last axis.
    left, right : scalar
        The support of f. f is assumed to be zero outside of it and not
        evaluated.
    
    Return
    ------
    out : array (M, K)
        The cross correlation evaluated on tout.
    """
    out = np.zeros(tout.shape)
    
    for iouter in numba.prange(len(out)):
        td = t[iouter]
        tg = tout[iouter]
        g = out[iouter]
        
        idr = 0
        igl = 0
        igr = 0
        for idl in range(len(td)):
            
            # Find idr such that the slice td[idl:idr] is contained in the
            # support length of f
            while idr < len(td) and td[idr] <= td[idl] + (right - left):
                idr += 1
            
            # Advance igl until the cross correlation at tg[igl] requires
            # just td[idl:idr]
            while igl < len(tg) and tg[igl] < td[idr - 1] - right:
                igl += 1
            
            # Advance igr until the cross correlation at tg[igr] may not
            # require just td[idl:idr]
            igr = igl
            while igr < len(tg) and tg[igr] <= td[idl] - left:
                igr += 1
            
            # Compute the cross correlation at tg[igl:igr]
            while igl < igr:
                for i in range(idl, idr):
                    g[igl] += f(td[i] - tg[igl])
                g[igl] /= len(td)
                # g[igl] = 1/len(td) * np.sum(f(td[idl:idr] - tg[igl]))
                igl += 1
    
    return out

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from scipy import interpolate
    
    fig = plt.figure('ccdelta1')
    fig.clf()
    
    ax = fig.subplots(1, 1)
    
    @numba.njit
    def f(t):
        return 1 if 0 < t < 1 else 0
    
    gen = np.random.default_rng(202012201928)
    t = np.sort(gen.uniform(0, 20, 100))
    tout = np.linspace(-2, 22, 10000)
    
    out = len(t) * ccdelta(f, t[None], tout[None], 0, 1)[0]
    
    y = interpolate.interp1d(tout, out)
    ax.plot(tout, out)
    ax.plot(t, y(t), 'x', color='black')
    ax.plot(t - 1, y(t - 1), '+', color='black')
    
    ax.grid()
    
    fig.tight_layout()
    fig.show()

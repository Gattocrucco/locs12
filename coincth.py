from scipy import special
import numpy as np

def coincrate(mincount, tcoinc, rate):
    """
    Compute the coincidence rate.
    
    Parameters
    ----------
    mincount : scalar or array
        The number of hits in the time window required to yield a coincidence.
    tcoinc : scalar or array
        The duration of the coincidence window.
    rate : scalar or array
        The rate of the poisson process.
    
    Return
    ------
    crate : scalar or array
        The rate of coincidences over unique but potentially overlapping sets
        of hits.
    """
    mu = rate * tcoinc
    mink = np.ceil(mincount)
    return rate * np.where(mink <= 1, mu > 0, special.gammainc(mink - 1, mu))

def _coincrate2(mincount, tcoinc, rate):
    """cross check of coincrate"""
    mu = rate * tcoinc
    mink = np.ceil(mincount)
    return rate * np.where(mink < 2, mu > 0, special.pdtrc(mink - 2, mu))

def coinctime(mincount, crate, rate):
    """
    Compute the coincidence time.
    
    Parameters
    ----------
    mincount : scalar or array
        The number of hits in the time window required to yield a coincidence.
    crate : scalar or array
        The required coincidence rate.
    rate : scalar or array
        The rate of the poisson process.
    
    Return
    ------
    tcoinc : scalar or array
        The duration of the coincidence window.
    """
    mink = np.ceil(mincount)
    return special.gammaincinv(np.maximum(0, mink - 1), crate / rate) / rate

def deadtimerate(rate, deadtime, restartable):
    """
    Compute the rate reduction with dead time.
    
    Parameters
    ----------
    rate : scalar or array
        The rate of the poisson process.
    deadtime : scalar or array
        The time after a hit under which other hits are discarded.
    restartable : bool
        If True, a discarded hit projects a dead time, otherwise not.
    
    Return
    ------
    drate : scalar or array
        The reduced rate of non-discarded hits.
    """
    mu = rate * deadtime
    if restartable:
        p = np.exp(-mu)
    else:
        p = 1 / (1 + mu)
        # this formula is empirical, it has the correct asymptotes:
        # mu -> 0       p ~ 1 - mu
        # mu -> inf     p ~ 1 / mu
    return rate * p

if __name__ == '__main__':
    
    t = 0.6123
    r = 1.9382
    for f in coincrate, _coincrate2:
        assert f(mincount=0, tcoinc=t, rate=r) == r
        assert f(1, t, r) == r
        assert f(0, 0, r) == 0
        assert f(0, t, 0) == 0
        assert f(1, 0, r) == 0
        assert f(1, t, 0) == 0
        assert f(2, 0, r) == 0
        assert f(2, t, 0) == 0
    
    assert coinctime(0, 0, 1) == 0
    assert coinctime(1, 0, 1) == 0
    assert np.isnan(coinctime(1, 0, np.array(0)))
    assert np.isinf(coinctime(1, 1, 1))
    assert np.isnan(coinctime(1, 1.1, 1))
    
    from matplotlib import pyplot as plt
    
    fig, ax = plt.subplots(num='coincth', clear=True)
    
    mincount = np.linspace(0, 10, 1001)
    for tcoinc in [0.5, 1, 2, 4]:
        y = coincrate(mincount, tcoinc, r)
        y2 = _coincrate2(mincount, tcoinc, r)
        assert np.allclose(y, y2)
        t = coinctime(mincount, y, r)
        assert np.allclose(np.where(mincount <= 1, tcoinc, t), tcoinc)
        ax.plot(mincount, y, label=f'f1, T={tcoinc}')
        ax.plot(mincount, y2, label=f'f2, T={tcoinc}', linestyle='--')
    
    ax.legend()
    
    fig.tight_layout()
    fig.show()

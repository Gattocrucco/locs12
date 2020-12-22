import numpy as np
from scipy import stats, special
from matplotlib import pyplot as plt
import numba
import numba_scipy_special

import ccdelta

def p_exp(t, tau):
    """
    Exponential pdf with scale tau.
    """
    return stats.expon.pdf(t, scale=tau)

def p_gauss(t, sigma):
    """
    Gaussian pdf with scale sigma.
    """
    return stats.norm.pdf(t, scale=sigma)

def p_S1(t, VL, tauV, tauL):
    """
    Mixture of two exponential pdfs "V" and "L", VL is the ratio of the
    coefficients V to L, tauV and tauL are the scale parameters.
    """
    V = 1 / (1 + 1 / VL)
    L = 1 / (1 + VL)
    return V * p_exp(t, tauV) + L * p_exp(t, tauL)

@numba.vectorize(nopython=True)
def log_p_exp_gauss(t, tau, sigma):
    """
    (Logarithm of) exponential pdf with scale tau convoluted with a normal with
    scale sigma.
    """
    return -np.log(tau) - t/tau + 1/2 * (sigma / tau) ** 2 + special.log_ndtr(t / sigma - sigma / tau)

@numba.njit
def p_exp_gauss(t, tau, sigma):
    """
    Exponential pdf with scale tau convoluted with a normal with scale sigma.
    """
    return np.exp(log_p_exp_gauss(t, tau, sigma))

@numba.njit
def p_S1_gauss(t, VL, tauV, tauL, sigma):
    """
    p_S1 convoluted with a normal with scale sigma.
    """
    V = 1 / (1 + 1 / VL)
    L = 1 / (1 + VL)
    return V * p_exp_gauss(t, tauV, sigma) + L * p_exp_gauss(t, tauL, sigma)

def check_p_S1(VL=3, tauV=7, tauL=1600, tres=10):
    """
    plot p_S1 and p_S1_gauss.
    """
    fig = plt.figure('temps1.check_p_S1')
    fig.clf()
    
    ax = fig.subplots(1, 1)
    
    ax.set_title('Time distribution of S1 photons')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Probabilty density [ns$^{-1}$]')
    
    t = np.linspace(-3 * tres, tauL, 10000)
    y = p_S1(t, VL, tauV, tauL)
    yc = p_S1_gauss(t, VL, tauV, tauL, tres)
    yg = p_gauss(t, tres)
    
    ax.plot(t, y, label='S1')
    ax.plot(t, yc, label='S1 + res')
    ax.plot(t, yg, label='res')
    
    ax.legend(loc='best')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')
    ax.set_yscale('log')
    ax.set_ylim(np.min(yc), np.max(y))

    fig.tight_layout()
    fig.show()
    
    return fig

def gen_S1(size, VL, tauV, tauL, tres, generator=None):
    """
    Generate samples from the pdf p_S1_gauss.
    """
    size = (size,) if np.isscalar(size) else tuple(size)
    if generator is None:
        generator = np.random.default_rng()
    
    expsamp = generator.standard_exponential(size=size + (2,))
    expsamp *= [tauV, tauL]
    choice = generator.binomial(n=1, p=1 / (1 + VL), size=size)
    ogrid_indices = tuple(slice(s) for s in size)
    indices = tuple(np.ogrid[ogrid_indices])
    samp = expsamp[indices + (choice,)]
    assert samp.shape == size
    
    normsamp = generator.standard_normal(size=size)
    normsamp *= tres
    samp += normsamp
    
    return samp

def check_gen_S1(VL=3, tauV=10, tauL=100, tres=5):
    """
    Check gen_S1.
    """
    fig = plt.figure('temps1.check_gen_S1')
    fig.clf()
    
    ax = fig.subplots(1, 1)
    
    ax.set_title('Time distribution of S1 photons')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Probabilty density [ns$^{-1}$]')

    gen = np.random.default_rng(202012181733)
    s = gen_S1((100, 100, 10), VL, tauV, tauL, tres, gen).reshape(-1)
    left, right = np.quantile(s, [0.001, 0.999])
    s = s[(s >= left) & (s <= right)]
    
    ax.hist(s, bins='auto', histtype='step', density=True, label='samples histogram')
    
    t = np.linspace(np.min(s), np.max(s), 10000)
    y = p_S1_gauss(t, VL, tauV, tauL, tres)
    
    ax.plot(t, y, label='pdf')
    
    ax.legend(loc='best')
    ax.grid()
    ax.set_yscale('log')
    
    fig.tight_layout()
    fig.show()
    
    return fig

def gen_DCR(size, T, rate, generator=None):
    """
    Generate uniformly distributed hits. The output shape is size + number of
    events per time window T.
    """
    size = (size,) if np.isscalar(size) else tuple(size)
    if generator is None:
        generator = np.random.default_rng()
    size += (int(np.rint(T * rate)),)
    return generator.uniform(0, T, size)

def filter_sample_mode(thit):
    """
    Compute the inverse of the time interval between consecutive hits.
    
    Parameters
    ----------
    thit : array (nevents, nhits)
        The input hit times. Each event must be already sorted.
    
    Return
    ------
    out : array (nevents, nhits - 1)
        The filter output, computed at the central point between consecutive
        hits.
    """
    return 1 / np.diff(thit, axis=-1)

def filter_cross_correlation(thit, tout, fun, left, right):
    """
    Cross-correlate a function with the temporal hits and compute it at given
    points. The output is:
    
        g(t) = 1/nhits * sum_i f(t_i - t)
    
    Parameters
    ----------
    thit : array (nevents, nhits)
        The input hit times. Each event must be already sorted.
    tout : array (nevents, nout)
        The times where the filter output is computed.
    fun : function
        A jittable function with signature scalar -> scalar.
    left, right : scalar
        Support of the function.
    
    Return
    ------
    out : array (nevents, nout)
        The filter output.
    """
    return ccdelta.ccdelta(fun, thit, tout, left, right)

def filter_sample_mode_cross_correlation(thit, tout, fun, left, right):
    """
    Cross-correlate a function with the inverse of the time interval between
    consecutive hits and compute it at given points. The output is:
    
        g(t) = sum_i f((t_i + t_i+1) / 2 - t) / (t_i+1 - t_i)
    
    Parameters
    ----------
    thit : array (nevents, nhits)
        The input hit times. Each event must be already sorted.
    tout : array (nevents, nout)
        The times where the filter output is computed.
    fun : function
        A ufunc with signature f(scalar) -> scalar.
    left, right : scalar
        Support of the function.
    
    Return
    ------
    out : array (nevents, nout)
        The filter output.
    """
    density = 1 / np.diff(thit, axis=-1)
    center = (thit[:, 1:] + thit[:, :-1]) / 2
    return ccdelta.ccdelta(fun, center, tout, left, right, w=density)

def check_filters(nsignal=100, T=4e6, rate=0.0025, VL=3, tauV=7, tauL=1600, tres=10):
    """
    Plot filters output.
    """
    generator = np.random.default_rng(202012191535)

    signal_loc = T / 2
    hits1 = gen_S1(nsignal, VL, tauV, tauL, tres, generator)
    hitdcr = gen_DCR((), T, rate, generator) - signal_loc
    hitall = np.concatenate([hits1, hitdcr])
    
    things = [
        ['signal only', hits1],
        ['noise only', hitdcr],
        ['all hits', hitall]
    ]
    figs = []
    
    for i, (desc, hits) in enumerate(things):
        figtitle = f'temps1.check_filters_{desc.replace(" ", "_")}'
        fig = plt.figure(figtitle, figsize=[10.72,  7.05])
        fig.clf()
    
        ax = fig.subplots(1, 1)
        
        hits = np.sort(hits)

        fed = filter_sample_mode(hits[None, :])[0]
        t_fed = (hits[1:] + hits[:-1]) / 2
        
        fun = lambda t: p_S1_gauss(t, VL, tauV, tauL, tres)
        left = -5 * tres
        right = 10 * tauL
        
        margin = 3 * tauL
        t = np.concatenate([hits[:1] - margin, hits, hits[-1:] + margin])
        t = t[:-1, None] + np.arange(5) / 5 * np.diff(t)[:, None]
        t = t.reshape(-1)
        
        fcc = filter_cross_correlation(hits[None, :], t[None, :], fun, left, right)[0]
        fedcc = filter_sample_mode_cross_correlation(hits[None, :], t[None, :], fun, left, right)[0]
    
        kw = dict(alpha=0.5)
        ax.plot(t_fed, fed / np.max(fed), label='sample mode', **kw)
        ax.plot(t, fcc / np.max(fcc), label='cross correlation', **kw)
        ax.plot(t, fedcc / np.max(fedcc), label='sample mode cross correlation', **kw)
        ax.plot(hits, np.full_like(hits, 1), '.k', markersize=2)
    
        ax.set_title(desc.capitalize())
        if ax.is_last_row():
            ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Filter output')
        
        ax.legend(loc='best', fontsize='small')
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
        ax.set_yscale('log')

        fig.tight_layout()
        figs.append(fig)
    
    for fig in figs:
        fig.show()
    
    return tuple(figs)

def all_filters(hits, VL, tauV, tauL, tres, midpoints=1):
    """
    Run all filters on hit times.
    
    Parameters
    ----------
    hits : array (nevents, nhits)
        The hit times. Need not be sorted.
    VL, tauV, tauL, tres : scalar
        p_S1_gauss parameters.
    midpoints : int
        The continuous filters are computed on the hits times and on
        `midpoints` evenly spaced intermediate points between each hit.
    
    Return
    ------
    out : array (nevents,)
        Structured numpy array with each field corresponding to a filter.
        The data type of each field is itself structured with fields 'max' and
        'maxtime' which contain the maximum filter value and its temporal
        location.
    """
    hits = np.sort(hits, axis=-1)

    fed = filter_sample_mode(hits)
    t_fed = (hits[:, 1:] + hits[:, :-1]) / 2
    
    # fun = lambda t: p_S1_gauss(t, VL, tauV, tauL, tres)
    # left = -5 * tres
    # right = 10 * tauL
    #
    # t = hits[:, :-1, None] + np.arange(midpoints + 1) / (midpoints + 1) * np.diff(hits, axis=-1)[:, :, None]
    # t = t.reshape(hits.shape[0], (hits.shape[1] - 1) * (midpoints + 1))
    #
    # fcc = filter_cross_correlation(hits, t, fun, left, right)
    # fedcc = filter_sample_mode_cross_correlation(hits, t, fun, left, right)
    
    out = np.empty(len(hits), dtype=[
        (filter_name, [('max', float), ('maxtime', float)])
        for filter_name in [
            'sample mode',
            # 'cross correlation',
            # 'sample mode cross correlation'
        ]
    ])
    
    # for t, f, n in zip([t_fed, t, t], [fed, fcc, fedcc], out.dtype.names):
    for t, f, n in [(t_fed, fed, out.dtype.names[0])]:
        out[n]['max'] = np.max(f, axis=-1)
        out[n]['maxtime'] = t[np.arange(len(hits)), np.argmax(f, axis=-1)]
    
    return out

def simulation(
    DCR=250e-9,  # (ns^-1) Dark count rate per PDM, 25 or 250 Hz
    VL=3,        # fast/slow ratio, ER=0.3, NR=3
    tauV=7,      # (ns) fast component tau
    tauL=1600,   # (ns) slow component tau
    T=4e6,       # (ns) time window (4 ms)
    npdm=8280,   # number of PDMs
    nphotons=10, # (2-100) number of photons in the S1 signal
    tres=3,      # (ns) temporal resolution (3-10)
    nmc=100,     # number of simulated events
):
    generator = np.random.default_rng(202012191535)

    hits1 = gen_S1((nmc, nphotons), VL, tauV, tauL, tres, generator)
    hitdcr = gen_DCR(nmc, T, DCR * npdm, generator)

    s1loc = T / 2
    hitall = np.concatenate([hits1 + s1loc, hitdcr], axis=-1)

    midpoints = 1
    fall = all_filters(hitall, VL, tauV, tauL, tres, midpoints)
    fdcr = all_filters(hitdcr, VL, tauV, tauL, tres, midpoints)
    
    figs = []
    
    for fname in fdcr.dtype.names:

        fig = plt.figure('temps1.simulation_' + fname.replace(" ", "_"))
        fig.clf()
        ax = fig.subplots(1, 1)
    
        ax.set_title(f'S1 localization with temporal information\n{fname.capitalize()} filter')
        ax.set_xlabel('Threshold on peak filter output')
    
        ax.plot(np.sort(fdcr[fname]['max']), np.linspace(1, 0, nmc), drawstyle='steps-pre', label='False positives (Fake S1/No S1 events)')
    
        s1tol = tauL
        close = np.abs(fall[fname]['maxtime'] - s1loc) < s1tol
        
        fmax = fall[fname]['max']
        isort = np.argsort(fmax)
        fmax = fmax[isort]
        s1 = close[isort]
        
        trues1 = np.cumsum(s1[::-1])[::-1]
        fakes1 = np.cumsum(~s1[::-1])[::-1]
        
        ax.plot(fmax, trues1 / nmc, label='Sensitivity (True S1/S1 events)', drawstyle='steps-pre')
        ax.plot(fmax, fakes1 / nmc, label='Mismatch (Fake S1/S1 events)', drawstyle='steps-pre')
    
        ax.set_xscale('log')
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
        ax.set_ylim(0, 1)
        ax.legend(loc='best')
        
        info = f"""\
DCR = {DCR * npdm * 1e3:.2g} $\\mu$s$^{{-1}}$
T = {T * 1e-6:.1f} ms
fast/slow = {VL:.1f}
nphotons = {nphotons}
temporal res. = {tres:.1f} ns"""

        kw = dict(
            xytext=(8, 8),
            va='bottom',
            ha='left',
            fontsize='small',
            xycoords='axes fraction',
            textcoords='offset points',
            bbox=dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='gray',
                boxstyle='round'
            ),
        )
        
        ax.annotate(info, (0, 0), **kw)    
        
        fig.tight_layout()
        figs.append(fig)
    
    for fig in figs:
        fig.show()
    
    return figs

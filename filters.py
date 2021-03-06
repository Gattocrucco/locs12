"""
Module with filters for hit times series.
"""

import collections

import numpy as np
from matplotlib import pyplot as plt
import numba

import pS1
import ccdelta
import runsliced
import dcr

def filter_sample_mode(thit):
    """
    Compute the logarithm of the inverse of the time interval between
    consecutive hits.
    
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
    return -np.log(np.diff(thit, axis=-1))

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
    density = -np.log(np.diff(thit, axis=-1))
    center = (thit[:, 1:] + thit[:, :-1]) / 2
    return ccdelta.ccdelta(fun, center, tout, left, right, w=density)

def addmidpoints(hits, midpoints):
    if midpoints == 0:
        return hits
    t = hits[:, :-1, None] + np.arange(midpoints + 1) / (midpoints + 1) * np.diff(hits, axis=-1)[:, :, None]
    t = t.reshape(hits.shape[0], (hits.shape[1] - 1) * (midpoints + 1))
    t = np.concatenate([t, hits[:, -1:]], axis=-1)
    return t

def filters(
    hits,
    VL,
    tauV,
    tauL,
    tres,
    midpoints=1,
    which=['sample mode', 'cross correlation'],
    pbar_batch=None,
    VLER=0.3,
    VLNR=3,
    tcoinc=200,
    dcr=None,
    nph=None,
    sigma=None,
):
    """
    Run filters on hit times.
    
    Parameters
    ----------
    hits : array (nevents, nhits)
        The hit times. Need not be sorted.
    VL : scalar
        VL p_S1_gauss parameter for the filters 'cross correlation',
        'likelihood', 'sample mode cross correlation'.
    tauV, tauL, tres : scalar
        p_S1_gauss parameters for the filters 'cross correlation', 'ER', 'NR',
        'fast', 'slow', 'likelihood', 'sample mode cross correlation'.
    midpoints : int
        The continuous filters are computed on the hits times and on
        `midpoints` evenly spaced intermediate points between each hit, apart
        from the 'coinc*' filters which are computed only on hits.
    which : list of strings
        The filters to compute. Keywords:
            'cross correlation'
            'ER'
            'NR'
            'fast'
            'slow'
            'coinc'
            'coincX' where X is the time in nanoseconds
            'likelihood'
            'sample mode'
            'sample mode cross correlation'
    pbar_batch : int, optional
        If given, a progressbar is shown that ticks every `pbar_batch` events.
    VLER, VLNR : scalar
        VL p_S1_gauss parameter for the ER and NR filters.
    tcoinc : scalar
        Time for the 'coinc' filter.
    dcr, nph, sigma: scalar
        Parameters for the 'likelihood' filter.
    
    Return
    ------
    out : array (nevents,)
        Structured numpy array with each field corresponding to a filter.
        The field values are themselves structured with fields 'time' and
        'value' containing arrays with the filter output.
    """
    hits = np.asarray(hits)
    assert len(hits.shape) == 2
    
    nt = (hits.shape[1] - 1) * (midpoints + 1) + 1
    def filtlength(f):
        if f == 'sample mode':
            return hits.shape[1] - 1
        elif f.startswith('coinc'):
            return hits.shape[1]
        else:
            return nt
    
    out = np.empty(len(hits), dtype=[
        (filter_name, [
            ('time', float, (filtlength(filter_name),)),
            ('value', float, (filtlength(filter_name),))
        ]) for filter_name in which
    ])
    
    all_hits = hits
    all_out = out
    
    template = dict()
    
    smcc = 'sample mode cross correlation'
    havesmcc = smcc in which
    
    sm = 'sample mode'
    havesm = sm in which

    for vl, f in [(VL, smcc), (VL, 'cross correlation'), (VLER, 'ER'), (VLNR, 'NR')]:
        if f not in which:
            continue
        offset = pS1.p_S1_gauss_maximum(vl, tauV, tauL, tres)
        ampl = pS1.p_S1_gauss(offset, vl, tauV, tauL, tres)
        fun = lambda t: pS1.p_S1_gauss(t + offset, vl, tauV, tauL, tres) / ampl
        template[f] = (numba.njit('f8(f8)')(fun), -5 * tres, 10 * tauL)
    
    for tau, f in [(tauV, 'fast'), (tauL, 'slow')]:
        if f not in which:
            continue
        offset = pS1.p_exp_gauss_maximum(tau, tres)
        ampl = pS1.p_exp_gauss(offset, tau, tres)
        fun = lambda t: pS1.p_exp_gauss(t + offset, tau, tres) / ampl
        template[f] = (numba.njit('f8(f8)')(fun), -5 * tres, max(10 * tau, 5 * tres))
    
    coincbounds = {}
    for f in which:
        if f.startswith('coinc'):
            T = float(f[5:]) if len(f) > 5 else tcoinc
            eps = 1e-6
            coincbounds[f] = (-eps * T, (1 - eps) * T)
    if coincbounds:
        templcoinc = numba.njit('f8(f8)')(lambda t: 1)
    
    if 'likelihood' in which:
        offset = pS1.p_S1_gauss_maximum(vl, tauV, tauL, sigma)
        ampl = pS1.log_likelihood(offset, vl, tauV, tauL, sigma, dcr, nph)
        fun = lambda t: pS1.log_likelihood(t + offset, vl, tauV, tauL, sigma, dcr, nph) / ampl
        template['likelihood'] = (numba.njit('f8(f8)')(fun), -5 * sigma, 10 * tauL)

    def batch(s):
        hits = all_hits[s]
        out = all_out[s]
        
        hits = np.sort(hits, axis=-1)
    
        timevalue = {}
    
        if havesm:
            v = filter_sample_mode(hits)
            t = (hits[:, 1:] + hits[:, :-1]) / 2
            timevalue[sm] = (t, v)
        
        if havesmcc or template:
            t = addmidpoints(hits, midpoints)
    
        for filt, (fun, left, right) in template.items():
            v = filter_cross_correlation(hits, t, fun, left, right)
            timevalue[filt] = (t, v)
        
        if havesmcc:
            fun, left, right = template[smcc]
            v = filter_sample_mode_cross_correlation(hits, t, fun, left, right)
            timevalue[smcc] = (t, v)
    
        for filt, (left, right) in coincbounds.items():
            v = filter_cross_correlation(hits, hits, templcoinc, left, right)
            timevalue[filt] = (hits, v)
        
        for k, (t, v) in timevalue.items():
            out[k]['time'] = t
            out[k]['value'] = v
    
    runsliced.runsliced(batch, len(out), pbar_batch)
    
    return out

def check_filters(nsignal=100, T=1e5, rate=0.0025, VL=3, tauV=7, tauL=1600, tres=10):
    """
    Plot filters output.
    """
    generator = np.random.default_rng(202012191535)

    signal_loc = T / 2
    hits1 = pS1.gen_S1(nsignal, VL, tauV, tauL, tres, generator)
    hitdcr = dcr.gen_DCR((), T, rate, generator) - signal_loc
    hitall = np.concatenate([hits1, hitdcr])
    
    things = [
        # ['signal only', hits1],
        ['noise only', hitdcr],
        ['all hits', hitall]
    ]
    figs = []
    
    for i, (desc, hits) in enumerate(things):
        figtitle = f'filters.check_filters_{desc.replace(" ", "_")}'
        fig, ax = plt.subplots(num=figtitle, clear=True, figsize=[10.72, 7.05])
        
        out = filters(hits[None], VL, tauV, tauL, tres, midpoints=4, which=[
            'sample mode',
            'cross correlation',
            'sample mode cross correlation'
        ])
        out = out[0]
    
        kw = dict(alpha=0.5)
        for k in out.dtype.names:
            t = out[k]['time']
            v = out[k]['value']
            ax.plot(t, v / np.max(v), label=k, **kw)
        ax.plot(hits, np.full_like(hits, 1), '.k', markersize=2)
    
        ax.set_title(desc.capitalize())
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Filter output (maximum=1)')
        
        ax.legend(loc='best', fontsize='small')
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')

        fig.tight_layout()
        figs.append(fig)
    
    for fig in figs:
        fig.show()
    
    return tuple(figs)

if __name__ == '__main__':
    check_filters()

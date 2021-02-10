import numpy as np
from matplotlib import pyplot as plt
import numba
from scipy import interpolate, signal

import pS1
import dcr
import filters
import textbox

def closenb(ref, nb, assume_sorted=False):
    """
    Return the array of elements from nb which are the closest to each element
    of ref.
    
    Parameters
    ----------
    ref : array (...)
        The reference values.
    nb : scalar or 1D array
        The neighbors.
    assume_sorted : bool
        If True the nb array is assumed to be sorted. Default False.
    
    Return
    ------
    out : int array (...)
        Array with the same shape of ref containing indices of elements of nb.
    """
    if not assume_sorted:
        nb = np.sort(nb)
    pos = np.searchsorted(nb, ref)
    posm1 = np.maximum(pos - 1, 0)
    pos = np.minimum(pos, len(nb) - 1)
    nbl = nb[posm1]
    nbh = nb[pos]
    assert np.all(ref - nbl) >= 0
    assert np.all(nbh - ref) >= 0
    return np.where(ref - nbl < nbh - ref, posm1, pos)

def hist(ax, x, **kw):
    counts, bins = np.histogram(x, bins='auto')
    counts = counts / np.max(counts)
    return ax.plot(np.concatenate([bins[:1], bins]), np.concatenate([[0], counts, [0]]), drawstyle='steps-post', **kw)

class TestCCFilter:

    def __init__(self, nevents=1, nsignal=10, T=10000, rate=0.0025, VL=3, tauV=7, tauL=1600, tres=10, VLfilter=None, dt=1, offset=0, seed=None, midpoints=1):
        if VLfilter is None:
            VLfilter = VL
    
        if seed is None:
            seedgen = np.random.default_rng()
            seed = seedgen.integers(10001)
        generator = np.random.default_rng(seed)

        hits1 = pS1.gen_S1((nevents, nsignal), VL, tauV, tauL, tres, generator)
        signal_loc = (T / 10 + 5 * tres) + T * np.arange(nevents)
        hits1 += signal_loc[:, None]
        hits1 = hits1.reshape(-1)
        hitdcr = dcr.gen_DCR((), T * nevents, rate, generator)
    
        hits = np.sort(np.concatenate([hits1, hitdcr]))
        taux = hits[:-1] + np.diff(hits) * np.arange(1, midpoints + 1)[:, None] / (midpoints + 1)
        taux = taux.reshape(-1)
        assert len(taux) == midpoints * (len(hits) - 1)
        points = np.sort(np.concatenate([hits, taux]))
    
        mx = pS1.p_S1_gauss_maximum(VL, tauV, tauL, tres)
        ampl = pS1.p_S1_gauss(mx, VL, tauV, tauL, tres)
        fun = numba.njit('f8(f8)')(lambda t: pS1.p_S1_gauss(t + mx + offset, VL, tauV, tauL, tres) / ampl)
        left = -5 * tres
        right = 10 * tauL
    
        t = np.arange(0, nevents * T, dt)

        v = filters.filter_cross_correlation(hits[None], t[None], fun, left, right)[0]
        interp = interpolate.interp1d(t, v)
        pidx, _ = signal.find_peaks(v, height=0.2)
        tpeak = t[pidx]
        hpeak = v[pidx]
        hidx = closenb(tpeak, points)
        thit = points[hidx]
        
        signal_loc_eff = signal_loc + mx + offset
        s1idx = closenb(signal_loc_eff, tpeak)
        s1 = np.zeros(len(tpeak), bool)
        s1[s1idx] = True
        
        self.hits1 = hits1      # S1 photons time
        self.hitdcr = hitdcr    # dark count photons time
        self.taux = taux        # additional times when the filter is evaluated
        self.t = t              # time where the filter is evaluated
        self.v = v              # filter output
        self.interp = interp    # interpolation of filter output
        self.tpeak = tpeak      # times of filter output peaks
        self.hpeak = hpeak      # height of filter output peaks
        self.thit = thit        # photons close to peaks
        self.s1 = s1            # mask for S1 peaks
        self.mx = mx            # point of maximum of p_S1_gauss
        self.signalloc = signal_loc_eff
        
        self.nevents = nevents
        self.nsignal = nsignal
        self.T = T
        self.rate = rate
        self.VL = VL
        self.tauV = tauV
        self.tauL = tauL
        self.tres = tres
        self.VLfilter = VLfilter
        self.dt = dt
        self.offset = offset
        self.seed = seed
        self.midpoints = midpoints
    
    @property
    def info(self):
        return f"""\
nevents = {self.nevents}
nsignal = {self.nsignal}
T = {self.T}
rate = {self.rate}
VL = {self.VL}
tauV = {self.tauV}
tauL = {self.tauL}
tres = {self.tres}
VL filter = {self.VLfilter}
dt = {self.dt}
offset = {self.mx:.2g} + {self.offset:.2g}
seed = {self.seed}
midpoints = {self.midpoints}"""
    
    def eventswhere(self, cond, which='signal'):
        """
        Select the events that contain at least a peak satisfying a given
        condition.
        
        Parameters
        ----------
        cond : str
            The condition. A numpy expression using the following variables:
                tpeak   time of filter output peak
                hpeak   height of filter output peak
                thit    time of hit closest to peak
                hhit    height of hit closest to peak
        which : {'signal', 'noise', 'both'}
            Apply the condition only to S1 peaks (the filter output peak
            closest to the S1 location), other peaks, or all peaks.
        
        Return
        ------
        events : int array
            Array of indices of events.
        """
        variables = dict(
            tpeak = self.tpeak,
            hpeak = self.hpeak,
            thit = self.thit,
            hhit = self.interp(self.thit),
        )
        expr = eval(cond, variables)
        expr = np.array(expr, bool)
        eventid = self.tpeak // self.T
        selection = dict(
            signal = self.s1,
            noise = ~self.s1,
            both = np.ones(len(self.tpeak), bool),
        )[which]
        return np.unique(eventid[expr & selection])
    
    def plotevent(self, event=0, zoomsignal=False):
        """
        Plot an event.
        
        Parameters
        ----------
        event : int
            Event index (0-based).
        zoomsignal : bool
            If True, zoom the plot around the signal location.
        
        Return
        ------
        fig : matplotlib figure
            The figure where the plot is drawn.
        """
        assert 0 <= event < self.nevents
        
        fig, ax = plt.subplots(num='testccfilter.TestCCFilter.plotevent', clear=True)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Filter output')

        L = max(0, event - 1) * self.T
        R = min(self.nevents, event + 2) * self.T
        
        sel = (L <= self.tpeak) & (self.tpeak <= R)
        ax.plot(self.tpeak[sel], self.hpeak[sel], 'o', color='#f55', label='peaks')
        textbox.textbox(ax, self.info, loc='upper right', fontsize='small')

        sel = (L <= self.t) & (self.t <= R)
        ax.plot(self.t[sel], self.v[sel], label='filter')
        
        sel = (L <= self.hits1) & (self.hits1 <= R)
        ax.plot(self.hits1[sel], self.interp(self.hits1[sel]), '.k', label='signal hits')
        
        sel = (L <= self.hitdcr) & (self.hitdcr <= R)
        ax.plot(self.hitdcr[sel], self.interp(self.hitdcr[sel]), 'xk', label='noise hits')
        
        if len(self.taux) > 0:
            sel = (L <= self.taux) & (self.taux <= R)
            ax.plot(self.taux[sel], self.interp(self.taux[sel]), '+k', label='auxiliary')

        sel = (L <= self.signalloc) & (self.signalloc <= R)
        kw = dict(label='signal location')
        for x in self.signalloc[sel]:
            ax.axvline(x, color='black', linestyle='--', **kw)
            kw.pop('label', None)
        
        l = event * self.T
        r = (event + 1) * self.T
        ax.axvspan(l, r, color='#ddd', label=f'event {event}')
    
        ax.legend(loc='upper center', fontsize='small', framealpha=0.9)
        
        if zoomsignal:
            xlim = self.signalloc[event] + np.array([-20 * self.tres, 60 * self.tauV])
        else:
            xlim = np.array([l, r]) + 0.1 * np.array([-1, 1]) * (r - l)
        ax.set_xlim(xlim)
        
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
    
        fig.tight_layout()
        
        return fig
    
    def plotdist(self):
        fig, axs = plt.subplots(3, 1, num='testccfilter.TestCCFilter.plotdist', clear=True, figsize=[6.4, 7.19])
    
        axs[0].set_xlabel('Time from peak to closest neighbor')
        axs[0].set_ylabel('Counts per bin [arb. un.]')

        axs[1].set_xlabel('Missing height to peak')
        axs[1].set_ylabel('Peak height')
    
        axs[2].set_xlabel('Missing height to peak')
        axs[2].set_ylabel('Counts per bin [arb. un.]')
        
        selkw = [
            (~self.s1, dict(label='noise')),
            ( self.s1, dict(label='signal'))
        ]
    
        for sel, kw in selkw:
            hist(axs[0], self.thit[sel] - self.tpeak[sel], **kw)
            missing = self.hpeak[sel] - self.interp(self.thit[sel])
            axs[1].plot(missing, self.hpeak[sel], '.', **kw)
            hist(axs[2], missing, **kw)
    
        textbox.textbox(axs[0], self.info, loc='upper left', fontsize='x-small')

        for ax in axs:
            ax.legend(loc='upper right', fontsize='medium')
            ax.minorticks_on()
            ax.grid(True, which='major', linestyle='--')
            ax.grid(True, which='minor', linestyle=':')
    
        fig.tight_layout()
        
        return fig

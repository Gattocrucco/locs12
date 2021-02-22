import numpy as np
from matplotlib import pyplot as plt
import numba
from scipy import interpolate, signal

import pS1
from dcr import gen_DCR
import filters
import textbox
import npzload

def closenb(ref, nb, assume_sorted=False):
    """
    Find the elements in nb which are the closest to each element of ref.
    
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
    """
    Plot a histogram of x on axis ax, with highest bin fixed to height 1.
    Keyword arguments are passed to ax.plot.
    """
    counts, bins = np.histogram(x, bins='auto')
    counts = counts / np.max(counts)
    return ax.plot(np.concatenate([bins[:1], bins]), np.concatenate([[0], counts, [0]]), drawstyle='steps-post', **kw)

def parabola(v, idx):
    """
    Interpolate minima/maxima with a parabola.
    
    Parameters
    ----------
    v : array (N,)
        Array of values at evenly spaced steps.
    idx : array (M,)
        Indices in v of local minima/maxima.
    
    Return
    ------
    dx : array (M,)
        The position of the extrema, relative to the indices in idx, in unit
        of 1 step.
    dy : array (M,)
        The height of the extrema relative to v[idx].
    """
    
    x0 = idx
    xp = np.minimum(x0 + 1, len(v))
    xm = np.maximum(x0 - 1, 0)
    
    y0 = v[x0]
    yp = v[xp]
    ym = v[xm]
    
    num = yp - ym
    denom = yp + ym - 2 * y0

    dx = -1/2 * num / denom
    dy = -1/8 * num ** 2 / denom
    
    return dx, dy
    
class TestCCFilter(npzload.NPZLoad):
    """
    Class to study where to compute the cross correlation filter.
    
    The description of the initialization parameters is in the code.
    
    Photon hits are simulated for a time nevents * T. Dark count photons are
    populated with the specified rate. Each T interval contains one S1 signal.
    The filter is evaluated on the whole sequence at steps of dt and
    interpolated quadratically. Local maxima are marked as peaks.
    
    The goal is to check if evaluating the filter only on hits and midpoints
    between consecutive hits is sufficient to find the local maxima without
    computing the whole filter output.
    
    Methods
    -------
    plotdist: plot distribution of peak alignment and missing height
    eventswhere: find events with peaks satisfying a condition
    plotevent: plot an event
    save: save an instance as npz archive
    
    Class methods
    -------------
    load: load a saved object from file
    
    Properties
    ----------
    interp: function to compute interpolated filter output
    info: human-readable description of simulation parameters
    midpoints: number of midpoints between hits (mutable)
    
    Example
    -------
    >>> sim = TestCCFilter(nevents=100)                          
    >>> sim.save('sim.npz')                                          
    >>> sim2 = TestCCFilter.load('sim.npz')             
    >>> sim2.plotdist().show()                                                
    >>> sim2.eventswhere('hpeak - hhit > 0.06')                               
    array([ 1, 64, 88, 96])
    >>> sim2.plotevent(64, zoomsignal=True).show() 
    >>> sim2.midpoints = 5 # try increasing the number of midpoints
    >>> sim2.plotdist().show                                 
    """

    def __init__(self,
        nevents=1,        # number of events
        nsignal=10,       # number of photons per S1
        T=10000,          # (ns) length of event
        rate=0.0025,      # (ns^-1) rate of dark count photons
        VL=3,             # fast/slow ratio of S1
        tauV=7,           # (ns) S1 fast tau
        tauL=1600,        # (ns) S1 slow tau
        tres=10,          # (ns) temporal resolution
        VLfilter=None,    # fast/slow ratio of filter, default same as VL
        dt=1,             # (ns) filter output sampling period
        offset=0,         # (ns) template is transformed as f'(t) = f(t+offset)
        seed=None,        # seed of random generator
        midpoints=1,      # number of points inserted between consecutive hits
        likelihood=False, # if True, use the likelihood instead of cckde
        dcr=None,         # DCR for the likelihood, default <rate>
        nph=None,         # S1 photons for the likelihood, default <nsignal>
        sigmakde=0,       # (ns) sigma of the KDE
    ):
        if VLfilter is None:
            VLfilter = VL
    
        if seed is None:
            seedgen = np.random.default_rng()
            seed = seedgen.integers(10001)
        generator = np.random.default_rng(seed)
        
        if dcr is None:
            dcr = rate
            
        if nph is None:
            nph = nsignal

        hits1 = pS1.gen_S1((nevents, nsignal), VL, tauV, tauL, tres, generator)
        signal_loc = (T / 10 + 5 * tres) + T * np.arange(nevents)
        hits1 += signal_loc[:, None]
        hits1 = hits1.reshape(-1)
        hitdcr = gen_DCR((), T * nevents, rate, generator)
    
        hits = np.sort(np.concatenate([hits1, hitdcr]))
        
        if likelihood:
            mx = pS1.p_S1_gauss_maximum(VLfilter, tauV, tauL, tres)
            ampl = pS1.log_likelihood(mx, VLfilter, tauV, tauL, tres, dcr, nph)
            fun = lambda t: pS1.log_likelihood(t + mx + offset, VLfilter, tauV, tauL, tres, dcr, nph) / ampl
            fun = numba.njit('f8(f8)')(fun)
        else:
            sigma = np.hypot(sigmakde, tres)
            mx = pS1.p_S1_gauss_maximum(VLfilter, tauV, tauL, sigma)
            ampl = pS1.p_S1_gauss(mx, VLfilter, tauV, tauL, sigma)
            fun = lambda t: pS1.p_S1_gauss(t + mx + offset, VLfilter, tauV, tauL, sigma) / ampl
            fun = numba.njit('f8(f8)')(fun)
        left = -5 * tres
        right = 10 * tauL
    
        t = np.arange(0, nevents * T, dt)
        v = filters.filter_cross_correlation(hits[None], t[None], fun, left, right)[0]
        
        pidx, _ = signal.find_peaks(v, height=0.9)
        dx, dy = parabola(v, pidx)
        tpeak = t[pidx] + dx * dt
        hpeak = v[pidx] + dy
        
        signal_loc_eff = signal_loc + mx + offset
        s1idx = closenb(signal_loc_eff, tpeak)
        s1 = np.zeros(len(tpeak), bool)
        s1[s1idx] = True
        
        self.hits1 = hits1       # S1 photons time
        self.hitdcr = hitdcr     # dark count photons time
        self.t = t               # time where the filter is evaluated
        self.v = v               # filter output
        self.tpeak = tpeak       # times of filter output peaks
        self.hpeak = hpeak       # height of filter output peaks
        self.s1 = s1             # mask for S1 peaks
        self.mx = mx             # point of maximum of p_S1_gauss
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
        self.likelihood = likelihood
        self.dcr = dcr
        self.nph = nph
        self.sigmakde = sigmakde
    
    @property
    def midpoints(self):
        """
        The number of midpoints between consecutive hits where the filter is
        evaluated. This property can be changed at any time.
        """
        return self._midpoints
    
    @midpoints.setter
    def midpoints(self, midpoints):
        hits = np.sort(np.concatenate([self.hits1, self.hitdcr]))
        taux = hits[:-1] + np.diff(hits) * np.arange(1, midpoints + 1)[:, None] / (midpoints + 1)
        taux = taux.reshape(-1)
        assert len(taux) == midpoints * (len(hits) - 1)
        points = np.sort(np.concatenate([hits, taux]))
    
        hidx = closenb(self.tpeak, points)
        thit = points[hidx]
    
        self.taux = taux    # additional times when the filter is evaluated
        self.thit = thit    # photons or midpoints close to peaks
        
        self._midpoints = midpoints
    
    @property
    def info(self):
        """
        Human-readable description of simulation parameters.
        """
        s = f"""\
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
offset = {self.offset} (+ {self.mx:.2g})
seed = {self.seed}
midpoints = {self.midpoints}
likelihood = {self.likelihood}"""
        
        if self.likelihood:
            s += f"""
dcr = {self.dcr}
nph = {self.nph}"""
        else:
            s += f"""
sigmakde = {self.sigmakde}"""
        
        return s

    @property
    def interp(self):
        """
        A quadratic interpolation of filter output.
        """
        kwargs = dict(assume_sorted=True, copy=False, kind='quadratic', bounds_error=False)
        return interpolate.interp1d(self.t, self.v, **kwargs)
    
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

        step = self.dt / 5
        t = np.arange(L, R + step / 2, step)
        ax.plot(t, self.interp(t), label='filter')
        
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
        """
        Plot some distributions relevant to the problem.
        
        Return
        ------
        fig : matplotlib figure
            The figure where the plot is drawn.
        """
        fig, axs = plt.subplots(2, 2, num='testccfilter.TestCCFilter.plotdist', clear=True, figsize=[8, 7], sharex='col')
    
        axs[0, 0].set_ylabel('Missing height to peak')
        axs[1, 0].set_ylabel('Counts per bin [arb. un.]')
        axs[1, 0].set_xlabel('Time from peak to closest neighbor')

        axs[0, 1].set_ylabel('Peak height')
        axs[1, 1].set_ylabel('Counts per bin [arb. un.]')
        axs[1, 1].set_xlabel('Missing height to peak')
        
        selkw = [
            (~self.s1, dict(label='noise')),
            ( self.s1, dict(label='signal'))
        ]
    
        for sel, kw in selkw:
            time = self.thit[sel] - self.tpeak[sel]
            height = self.hpeak[sel]
            missing = height - self.interp(self.thit[sel])
            axs[0, 0].plot(time, missing, '.', **kw)
            hist(axs[1, 0], time, **kw)
            axs[0, 1].plot(missing, height, '.', **kw)
            hist(axs[1, 1], missing, **kw)
    
        textbox.textbox(axs[1, 0], self.info, loc='upper left', fontsize='x-small')

        for ax in axs.reshape(-1):
            if ax.is_last_col():
                ax.legend(loc='upper right', fontsize='medium')
            ax.minorticks_on()
            ax.grid(True, which='major', linestyle='--')
            ax.grid(True, which='minor', linestyle=':')
    
        fig.tight_layout()
        
        return fig

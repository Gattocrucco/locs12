import collections

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
import numba

import pS1
import clusterargsort
import filters
import dcr
import textbox
import qsigma
import symloglocator
import npzload
import aligntwin

@numba.njit(cache=True)
def argmin_sliced(x, indices):
    """
    x is a 1d array. The argmin is done separately on each subarray
    x[indices[i]:indices[i+1]]. The returned indices are for the whole x array.
    """
    out = np.empty(len(indices) - 1, np.intp)
    for i in range(len(out)):
        out[i] = indices[i] + np.argmin(x[indices[i]:indices[i + 1]])
    return out

def plot_histogram(ax, counts, bins, **kw):
    """
    Plot the output of np.histogram (counts, bins) on axis ax. Keyword
    arguments are passed to ax.plot.
    """
    return ax.plot(np.concatenate([bins[:1], bins]), np.concatenate([[0], counts, [0]]), drawstyle='steps-post', **kw)

class Simulation(npzload.NPZLoad):
    """
    Class to simulate S1 photons and DCR with temporal information only.
    The simulation runs at object initialization.
    """
    
    def __init__(self,
        DCR=250e-9,      # (ns^-1) Dark count rate per PDM
        VL=3,            # fast/slow ratio, ER=0.3, NR=3
        tauV=7,          # (ns) fast component tau
        tauL=1600,       # (ns) slow component tau
        T=100e3,         # (ns) time per event
        npdm=8280,       # number of PDMs
        nphotons=10,     # number of photons in the S1 signal
        tres=10,         # (ns) temporal resolution
        sigma=None,      # (ns) sigma of filter template, default same as tres
        nmc=10,          # number of simulated events
        deadradius=2000, # (ns) for selecting S1 candidates in filter output
        generator=None,  # numpy random generator, or integer seed
        pbar_batch=10,   # number of events for each progress bar step
        filters=None,    # (list of) filters to use, default ER/NR cross corr.
        midpoints=3,     # no. of points between hits where the filter is eval.
    ):
        # Random generator.
        if generator is None:
            generator = np.random.default_rng()
        elif isinstance(generator, int):
            generator = np.random.default_rng(generator)
        
        # User-specified parameters.
        self.DCR = DCR
        self.VL = VL
        self.tauV = tauV
        self.tauL = tauL
        self.T = T
        self.npdm = npdm
        self.nphotons = nphotons
        self.tres = tres
        self.sigma = tres if sigma is None else sigma
        self.nmc = nmc
        self.deadradius = deadradius
        self.pbar_batch = pbar_batch
        if filters is None:
            filters = ['ER', 'NR']
        if isinstance(filters, str):
            filters = [filters]
        self.filters = np.array(filters)
        self.midpoints = midpoints
        
        # Internal parameters.
        self.s1loc = T / 2
        self.hitnames = np.array(['s1', 'dcr', 'all'])
        
        # Run the simulation.
        self._gen_photons(generator)
        self._run_filters()
        self._make_dicts()
        self._merge_candidates()
    
    @classmethod
    def load(cls, *args):
        self = super().load(*args)
        # these two functions save results to dictionaries which are not
        # saved to the npz archive.
        self._make_dicts()
        self._merge_candidates()
        return self
        
    def _gen_photons(self, generator):
        """
        Generate photons for S1, DCR, and merged, and save them to instance
        variables.
        """
        self.hits1 = self.s1loc + pS1.gen_S1((self.nmc, self.nphotons), self.VL, self.tauV, self.tauL, self.tres, generator)
        self.hitdcr = dcr.gen_DCR(self.nmc, self.T, self.DCR * self.npdm, generator)
        self.hitall = np.concatenate([self.hits1, self.hitdcr], axis=-1)
    
    def _run_filters(self):
        """
        Run filters on photon hit time series and save them to an instance
        variable.
        """
        for n in self.hitnames:
            hits = getattr(self, 'hit' + n)
            kw = dict(midpoints=self.midpoints, pbar_batch=self.pbar_batch, which=self.filters)
            f = filters.filters(hits, self.VL, self.tauV, self.tauL, self.sigma, **kw)
            setattr(self, 'filt' + n, f)
    
    def _make_dicts(self):
        """
        Put hit and filter output in dictionaries for convenience.
        """
        self.hitd = dict(s1=self.hits1, all=self.hitall, dcr=self.hitdcr)
        self.filtd = dict(s1=self.filts1, all=self.filtall, dcr=self.filtdcr)
        self.plotkw = {
            's1' : dict(label='One S1 events', linestyle='-', color='#0b0'),
            'all': dict(label='DCR + one S1 events', color='#00b', linestyle='--'),
            'dcr': dict(label='DCR events', linestyle=':', color='#b00'),
        }
    
    def _merge_candidates(self):
        """
        Take the output of filters, apply the dead radius, merge events,
        sort by filter output value, mark candidates considered true signal.
        """
        # dictionaries layout: filter name -> (photon series -> 1D array)
        self.times = collections.defaultdict(dict)
        self.values = collections.defaultdict(dict)
        self.signal = collections.defaultdict(dict)
        
        for k, fhits in self.filtd.items():
            for fname in fhits.dtype.names:
                
                time = fhits[fname]['time']
                value = fhits[fname]['value']
                
                indices1, length = clusterargsort.clusterargsort(value, time, self.deadradius)
                indices0 = np.repeat(np.arange(len(value)), np.diff(length))
    
                value = value[indices0, indices1]
                time = time[indices0, indices1]
                
                timedelta = np.abs(time - self.s1loc)
                minindices = argmin_sliced(timedelta, length)
                signal = np.zeros(len(time), bool)
                signal[minindices] = True
                # this will count signals even in dcr, just don't use them
    
                idx = np.argsort(value)
                
                self.times[fname][k] = time[idx]
                self.values[fname][k] = value[idx]
                self.signal[fname][k] = signal[idx]
    
    def _counts_interp(self, fname, signal=False):
        """
        Parameters:
        fname = filter name
        signal = if True, count just candidates marked as true S1
        Return:
        x : dictionary photon series -> sorted 1D array of threshold values
        interp : dictionary photon series -> (function scalar -> scalar)
        The function computes the number of candidates below a threshold,
        per event.
        The x threshold values are points where the function goes down by one
        step.
        """
        values = self.values[fname]
        if signal:
            values = {
                k: v[self.signal[fname][k]]
                for k, v in values.items()
            }
        
        x = dict(values)
        y = {
            k: (1 + np.arange(len(v)))[::-1] / self.nmc
            for k, v in values.items()
        }
        # x = threshold
        # y = number of candidates below threshold
        interpkw = dict(kind='next', assume_sorted=True, copy=False, bounds_error=False)
        interp = {
            k: interpolate.interp1d(x[k], y[k], fill_value=(y[k][0], 0), **interpkw)
            for k in values
        }
                
        return x, interp
        
    def candidates_above_threshold(self, fname, hits, signalonly=False, rate=False):
        """
        Return a function to compute the number of candidates above a given
        threshold, and the array of thresholds where the function has steps.
        
        Parameters
        ----------
        fname : str
            The filter name.
        hits : {'s1', 'dcr', 'all'}
            The photon selection. 's1' = only S1 photons, 'dcr' = only noise
            photons, 'all' = all photons together.
        signalonly : bool
            Default False. If True, count only true S1 signals. An error is
            raised if hits='dcr' and signalonly=True.
        rate : bool
            If True, compute the candidates per unit time (second) instead of
            per event. Default False.
        
        Return
        ------
        f : function scalar -> scalar
            A decreasing step function that maps a threshold to the number of
            candidates per event with filter output value >= that threshold.
        t : sorted 1D array
            The thresholds where f has steps.
        """
        assert not (signalonly and hits == 'dcr')
        
        cache = '_xinterp_s' if signalonly else '_xinterp'
        if not hasattr(self, cache):
            setattr(self, cache, {
                fname: self._counts_interp(fname, signalonly)
                for fname in self.values
            })
        
        x, interp = getattr(self, cache)[fname]
        x = x[hits]
        interp = interp[hits]
        
        if rate:
            factor = 1 / (self.T * 1e-9)
            interp0 = interp
            interp = lambda t: factor * interp0(t)
        
        return interp, x
    
    def _fname(self, fname):
        """
        Method to parse the fname parameter in plot methods.
        """
        if fname is not None:
            return fname
        elif len(self.filters) == 1:
            return self.filters[0]
        else:
            raise KeyError(fname)
    
    def efficiency_vs_rate(self, fname=None, signalhits='all'):
        """
        Give a function to compute the S1 detection efficiency given the rate
        of fake S1 in noise photons.
        
        Parameters
        ----------
        fname : str, optional
            The filter to use. Optional if there's only one filter.
        signalhits : {'all', 's1'}
            Whether to count signals within noise (default) or alone.
        
        Return
        ------
        f : function scalar -> scalar
            A piecewise linear function mapping fake rate to efficiency.
        r : sorted 1D array
            The rates where f changes slope.
        """
        fname = self._fname(fname)
        
        f,   t   = self.candidates_above_threshold(fname, 'dcr'     , signalonly=False, rate=True )
        fs1, ts1 = self.candidates_above_threshold(fname, signalhits, signalonly=True , rate=False)
        
        sel = (ts1[0] <= t) & (t <= ts1[-1])
        t = t[sel]
        
        t = np.sort(np.concatenate([t, ts1]))[::-1]

        r = np.concatenate([[0], f(t)])
        e = np.concatenate([[0], fs1(t)])
        interpkw = dict(kind='linear', assume_sorted=True, copy=False, bounds_error=False)
        f = interpolate.interp1d(r, e, fill_value=(e[0], e[-1]), **interpkw)
        return f, r
        
    def plot_filter_performance_threshold(self, fname=None):
        """
        Plot filter fake rate and efficiency vs threshold.
        
        Parameters
        ----------
        fname : str, optional
            The filter. Optional if there's only one filter.
        
        Return
        ------
        fig : matplotlib figure
            The figure where the plot is drawn.
        """
        fname = self._fname(fname)
        
        figname = 'temps1.Simulation.plot_filter_performance_threshold.' + fname.replace(" ", "_")
        fig, axs = plt.subplots(2, 1, num=figname, figsize=[6.4, 7.19], clear=True, sharex=True)
    
        ax = axs[0]
        ax.set_ylabel('Rate of S1 candidates [s$^{-1}$]')
        
        for k in ['all', 'dcr', 's1']:
            f, t = self.candidates_above_threshold(fname, k, rate=True)
            x = np.concatenate([t, t[-1:]])
            y = np.concatenate([f(t), [0]])
            ax.plot(x, y, drawstyle='steps-pre', **self.plotkw[k])
        
        rate1 = 1 / (self.T * 1e-9)
        ax.axhspan(0, rate1, color='#ddd', label='$\\leq$ 1 cand. per event')
    
        ax.set_yscale('log')
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
        ax.legend(loc='upper right', title=fname + ' filter')
    
    
        ax = axs[1]
        ax.set_ylabel('S1 detection efficiency')
        ax.set_xlabel('Threshold on filter output')
        
        for k in ['all', 's1']:
            f, t = self.candidates_above_threshold(fname, k, signalonly=True)
            x = np.concatenate([t, t[-1:]])
            y = np.concatenate([f(t), [0]])
            ax.plot(x, y, drawstyle='steps-pre', **self.plotkw[k])

        textbox.textbox(ax, self.infotext(), fontsize='small')
    
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
        ax.set_ylim(0, max(1, np.max(y)))
        
        fig.tight_layout()
        
        return fig
    
    def plot_filter_performance(self, filters=None):
        """
        Plot filter efficiency vs fake rate.
        
        Parameters
        ----------
        filters : (list of) str, optional
            The filters to plot. All filters if not specified.
        
        Return
        ------
        fig : matplotlib figure
            The figure where the plot is drawn.
        """
        if filters is None:
            filters = self.filters
        elif isinstance(filters, str):
            filters = [filters]
        
        figname = 'temps1.Simulation.plot_filter_performance'
        fig, ax = plt.subplots(num=figname, clear=True)
    
        ax.set_xlabel('Rate of S1 candidates in DCR [s$^{-1}$]')
        ax.set_ylabel('S1 detection efficiency')
        
        for i, fname in enumerate(filters):
            for k in ['all']:
                f, r = self.efficiency_vs_rate(fname, k)
                kw = dict(self.plotkw[k])
                kw.update(color=f'C{i}', label=fname + ' filter, ' + kw['label'])
                ax.plot(r, f(r), **kw)
        
        textbox.textbox(ax, self.infotext(), loc='upper left', fontsize='small')

        ax.legend(loc='lower right', fontsize='medium')
        ax.minorticks_on()
        ax.set_xscale('log')
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
        
        fig.tight_layout()
        
        return fig
    
    def plot_temporal_distribution(self, fname=None):
        """
        Plot the temporal distribution of S1 candidates.
        
        Parameters
        ----------
        fname : str, optional
            The filter. Optional if there's only one filter.
        
        Return
        ------
        fig : matplotlib figure
            The figure where the plot is drawn.
        """
        fname = self._fname(fname)
        
        figname = 'temps1.Simulation.plot_temporal_distribution.' + fname.replace(" ", "_")
        fig, axs = plt.subplots(2, 1, num=figname, clear=True, figsize=[6.4, 7.19])
        axs[0].set_title(f'{fname.capitalize()} filter\nTemporal distribution of S1 candidates')
    
        ax = axs[0]
        ax.set_xlabel('Time relative to true S1 location [ns]')
        ax.set_ylabel('Inverse of neighbor temporal gap [ns$^{-1}$]')
    
        for k in ['all', 'dcr']:
            time = self.times[fname][k]
            time = np.sort(time)
            ddecdf = 1 / np.diff(time)
            x = time - self.s1loc
            y = np.concatenate([ddecdf, ddecdf[-1:]])
            ax.plot(x, y, drawstyle='steps-post', **self.plotkw[k])
    
        ax.axvspan(-self.deadradius, self.deadradius, color='#eee', zorder=-9, label='$\\pm$ dead radius')

        ax.legend(loc='upper right')
        ax.set_xlim(3.5 * self.deadradius * np.array([-1, 1]))
        ax.set_yscale('log')
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
    
    
        ax = axs[1]
        ax.set_xlabel('Time relative to true S1 location [ns]')
        ax.set_ylabel('Histogram bin density [ns$^{-1}$]')
    
        times1 = self.hits1.reshape(-1) - self.s1loc               
        time = self.times[fname]['all'] - self.s1loc
        signal = self.signal[fname]['all']
        time_match = time[signal]
        sigma = qsigma.qsigma(time_match)
    
        # t = np.linspace(..., ..., 1000)
        # ax.plot(t, pS1.p_S1_gauss(t, self.VL, self.tauV, self.tauL, self.tres), label='S1 pdf')
        histkw = dict(bins='auto', density=True, histtype='step', zorder=10)
        ax.hist(times1, label=f'S1 photons ({len(times1)})', linestyle=':', **histkw)
        ax.hist(time_match, label=f'matching candidates ($\\sigma_q$={sigma:.3g})', **histkw)
    
        ax.axvspan(0, self.deadradius, color='#eee', zorder=-9, label='dead radius')

        textbox.textbox(ax, self.infotext(), loc='upper left', zorder=11)
    
        ax.legend(loc='upper right', fontsize='small')
        ax.set_yscale('log')
        linthreshx = 10 ** np.ceil(np.log10(15 * sigma))
        ax.set_xscale('symlog', linthreshx=linthreshx)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(symloglocator.MinorSymLogLocator(linthreshx))
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')

        fig.tight_layout()
        
        return fig

    def plot_filter_output_histogram(self, fname=None):
        """
        Plot the histogram of the filter peak value of S1 candidates for noise
        and signal.
        
        Parameters
        ----------
        fname : str, optional
            The filter. Optional if there's only one filter.
        
        Return
        ------
        fig : matplotlib figure
            The figure where the plot is drawn.
        """
        fname = self._fname(fname)
        
        figname = 'temps1.Simulation.plot_filter_output_histogram.'
        figname += fname.replace(' ', '_')
        fig, ax = plt.subplots(num=figname, clear=True)
        axr = ax.twinx()

        ax.set_title(fname.capitalize() + ' filter output distribution')
    
        ax.set_xlabel('Filter output value')
        ax.set_ylabel('Rate per bin [s$^{-1}$]')
        axr.set_ylabel('Fraction of S1 per bin [%]')
        
        x = self.values[fname]['dcr']
        counts, bins = np.histogram(x, bins='auto')
        counts = counts / (self.nmc * self.T * 1e-9)
        linenoise, = plot_histogram(ax, counts, bins, **self.plotkw['dcr'])

        x = self.values[fname]['s1'][self.signal[fname]['s1']]
        counts, bins = np.histogram(x, bins='auto')
        counts = counts * 100 / len(x)
        linesigpure, = plot_histogram(axr, counts, bins, **self.plotkw['s1'])
        N = len(x)
        
        x = self.values[fname]['all'][self.signal[fname]['all']]
        counts, bins = np.histogram(x, bins='auto')
        counts = counts * 100 / N
        linesig, = plot_histogram(axr, counts, bins, **self.plotkw['all'])
        
        textbox.textbox(axr, self.infotext(), loc='upper right')

        axr.legend([
            linenoise,
            linesigpure,
            linesig,
        ], [
            'Fake rate (left scale)',
            'Signal % (right scale)',
            'Signal within noise (relative)',
        ], loc='upper left')
        
        ax.minorticks_on()
        axr.minorticks_on()
        aligntwin.alignYaxes([ax, axr], [0, 0])
        ax.set_ylim(0, ax.get_ylim()[1])
        axr.set_ylim(0, axr.get_ylim()[1])
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
        
        fig.tight_layout()
        
        return fig

    def infotext(self):
        """
        Return a human-readable string with the values of the simulation
        parameters.
        """
        
        vlrange = [
            ('slow',  0  ,   0.05),
            ('ER'  ,  0.2,   0.4 ),
            ('NR'  ,  2.8,   3.2 ),
            ('fast', 20  , np.inf),
        ]
        
        for s, l, r in vlrange:
            if l <= self.VL <= r:
                svl = f' ({s})'
                break
        else:
            svl = ''
        
        return f"""\
nevents = {self.nmc}
T = {self.T * 1e-6:.1f} ms
total DCR = {self.DCR * self.npdm * 1e3:.2g} $\\mu$s$^{{-1}}$
fast/slow = {self.VL:.2g}{svl}
nphotons = {self.nphotons}
$\\tau$ = ({self.tauV:.1f}, {self.tauL:.0f}) ns
temporal res. = {self.tres:.2g} ns
sigma = {self.sigma:.2g} ns
dead radius = {self.deadradius:.0f} ns
midpoints = {self.midpoints}"""

if __name__ == '__main__':
    
    arguments = dict(
        DCR = [25e-9, 250e-9],
        VL = [0, 1e9],
        nphotons = [3, 5, 7, 10, 15, 20, 30],
        sigma = [5, 10, 15, 20, 25, 150, 170, 190, 210, 250]
    )
    fixedarguments = dict(
        nmc = 1000,
        pbar_batch = None,
        tres = 10,
        filters = ['fast', 'slow'],
    )
    outfile = 'temps1series0214.npy'
    
    ####################
    
    import warnings
    
    import tqdm
    from numpy.lib import format as nplf
    
    import named_cartesian_product
    
    warnings.filterwarnings("ignore")
    
    generator = np.random.default_rng(202102141242)
    
    arglist = named_cartesian_product.named_cartesian_product(**arguments)
    
    rate = np.concatenate([
        np.linspace(0, 100, 1000),
        np.logspace(np.log10(100), np.log10(2e6), 1000)[1:]
    ])
    threshold = np.concatenate([
        np.linspace(0, 1, 20)[:-1],
        np.logspace(np.log10(1), np.log10(np.max(arguments['nphotons'])), 1000)
    ])
    
    filterdtype = [
        ('efficiency', float, len(rate)),
        ('ratethr', float, len(threshold)),
        ('effthr', float, len(threshold)),
    ]
    table = nplf.open_memmap(outfile, mode='w+', shape=arglist.shape, dtype=[
        ('done', bool),
        ('parameters', arglist.dtype),
        ('rate', float, len(rate)),
        ('threshold', float, len(threshold)),
        ('fast', filterdtype),
        ('slow', filterdtype),
    ])
    table['done'] = False
    table['parameters'] = arglist
    table['rate'] = rate
    table['threshold'] = threshold
    table.flush()
    
    print(f'saving to {outfile}...')
    
    table0 = table
    table = table.view()
    table.shape = (table.size,)
    
    kw = dict(generator=generator)
    kw.update(fixedarguments)
    
    indices = np.arange(len(table))
    np.random.shuffle(indices)
    
    for i in tqdm.tqdm(indices):
        entry = table[i]
        
        argstruct = entry['parameters']
        argdict = {k: argstruct[k] for k in argstruct.dtype.names}
        kw.update(argdict)
        
        sim = Simulation(**kw)
        for fname in sim.filters:
            f, r = sim.efficiency_vs_rate(fname)
            entry[fname]['efficiency'] = f(rate)
            g, tg = sim.candidates_above_threshold(fname, 'dcr', False, True)
            entry[fname]['ratethr'] = g(threshold)
            h, th = sim.candidates_above_threshold(fname, 'all', True, False)
            entry[fname]['effthr'] = h(threshold)
        
        entry['done'] = True
        table0.flush()

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

import pS1
import clusterargsort
import filters
import dcr
import textbox
import qsigma
import symloglocator
import npzload

def clustersort(time, value, deadradius):
    """
    Given a series of filtered events, remove all points which are close to a
    point with higher value, unless the latter has already been removed by an
    even higher one.
    
    Parameters
    ----------
    time, value : array (nevents, npoints)
        The filter output.
    deadradius : scalar
        The temporal distance to consider points as close.
    
    Return
    ------
    time, value : array (N,)
        N <= nevents * npoints. The value array is sorted, with the time
        array matching the correct points in the value array. All events are
        merged together.
    """
    indices1, length = clusterargsort.clusterargsort(value, time, deadradius)
    indices0 = np.repeat(np.arange(len(value)), np.diff(length))
    
    value = value[indices0, indices1]
    time = time[indices0, indices1]
    
    idx = np.argsort(value)
    return time[idx], value[idx]

def autolinscale(ax, xratio=20, yratio=20):
    """
    If the x/y scale of a plot is logarithmic and the ratio between the high
    and low limits is below a specified ratio, set the scale to linear and the
    lower limit to 0.
    """

    l, h = ax.get_xlim()
    if ax.get_xscale() == 'log' and h / l <= xratio:
        ax.set_xscale('linear')
        ax.set_xlim(0, h)

    l, h = ax.get_ylim()
    if ax.get_yscale() == 'log' and h / l <= yratio:
        ax.set_yscale('linear')
        ax.set_ylim(0, h)

class Simulation(npzload.NPZLoad):
    
    def __init__(
        DCR=250e-9,      # (ns^-1) Dark count rate per PDM, 25 or 250 Hz
        VL=3,            # fast/slow ratio, ER=0.3, NR=3
        tauV=7,          # (ns) fast component tau
        tauL=1600,       # (ns) slow component tau
        T_target=4e6,    # (ns) time window
        T_sim=100e3,     # (ns) actually simulated time window
        npdm=8280,       # number of PDMs
        nphotons=10,     # (2-100) number of photons in the S1 signal
        tres=3,          # (ns) temporal resolution (3-10)
        nmc=10,          # number of simulated events
        deadradius=4000, # (ns) for selecting S1 candidates in filter output
        matchdist=2000,  # (ns) for matching a S1 candidate to the true S1
        generator=None,  # numpy random generator, or integer seed
        pbar_batch=None, # number of events for each progress bar step
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
        self.T_target = T_target
        self.T_sim = T_sim
        self.npdm = npdm
        self.nphotons = nphotons
        self.tres = tres
        self.nmc = nmc
        self.deadradius = deadradius
        self.matchdist = matchdist
        self.generator = generator
        self.pbar_batch = pbar_batch
        
        # Internal parameters.
        self.s1loc = T_sim / 2
    
    def infotext(self):
        """
        Return a human-readable string with the values of the simulation
        parameters.
        """
        return f"""\
total DCR = {self.DCR * self.npdm * 1e3:.2g} $\\mu$s$^{{-1}}$
T (target) = {self.T_target * 1e-6:.1f} ms
T (sim.) = {self.T_sim * 1e-6:.3f} ms
fast/slow = {self.VL:.1f}
nphotons = {self.nphotons}
$\\tau$ = ({self.tauV:.1f}, {self.tauL:.0f}) ns
temporal res. = {self.tres:.1f} ns
dead radius = {self.deadradius:.0f} ns
match dist. = {self.matchdist:.0f} ns
nevents = {self.nmc}"""

    def _gen_photons(self):
        self.hits1 = pS1.gen_S1((self.nmc, self.nphotons), self.VL, self.tauV, self.tauL, self.tres, self.generator)
        self.hitdcr = dcr.gen_DCR(self.nmc, self.T_sim, self.DCR * self.npdm, self.generator)
        self.hitall = np.concatenate([self.hits1 + self.s1loc, self.hitdcr], axis=-1)
    
        self.hitd = dict(s1=hits1, dcr=hitdcr, all=hitall)
        self.plotkw = {
            's1': dict(label='No DCR events')
            'all': dict(label='Single S1 events'),
            'dcr': dict(label='No S1 events', linestyle='--')
        }
    
    def _run_filters(self):
        self.filt = {
            k: filters.filters(hits, self.VL, self.tauV, self.tauL, self.tres, midpoints=1, pbar_batch=self.pbar_batch)
            for k, hits in self.hitd.items()
        }
    
        figs = []
    
        for fname in filt['all'].dtype.names:
        
            times = {}
            values = {}
            for k, fhits in filt.items():
                time = fhits[fname]['time']
                value = fhits[fname]['value']
                times[k], values[k] = clustersort(time, value, deadradius)
        
            figname = 'temps1.simulation_' + fname.replace(" ", "_")
            fig, axs = plt.subplots(2, 1, num=figname, figsize=[6.4, 7.19], clear=True, sharex=True)
            figs.append(fig)
            axs[0].set_title(f'{fname.capitalize()} filter detection performance\n(with explicit threshold)')
        
            ax = axs[0]
            ax.set_ylabel('Mean number of S1 candidates per event')
        
            xy = {
                k: [v, (1 + np.arange(len(v)))[::-1] / nmc]
                for k, v in values.items()
            }
            interpkw = dict(kind='next', assume_sorted=True, copy=False, bounds_error=False)
            interp = {
                k: interpolate.interp1d(x, y, fill_value=(y[0], 0), **interpkw)
                for k, (x, y) in xy.items()
            }
            x = np.sort(np.concatenate([xy['all'][0], xy['dcr'][0]]))
            xy['all'][0] = x
            xy['all'][1] = interp['all'](x) + interp['dcr'](x) * (T_target / T_sim - 1)
            xy['dcr'][1] *= T_target / T_sim
        
            for k, (x, y) in xy.items():
                x = np.concatenate([x, x[-1:]])
                y = np.concatenate([y, [0]])        
                ax.plot(x, y, drawstyle='steps-pre', **plotkw[k])
        
            if fname == 'sample mode':
                ax.set_xscale('log')
            ax.set_yscale('log')
            ax.minorticks_on()
            ax.grid(True, which='major', linestyle='--')
            ax.grid(True, which='minor', linestyle=':')
            ax.legend(loc='upper right')
        
        
            ax = axs[1]
            ax.set_ylabel('True S1 detection probability')
            ax.set_xlabel('Threshold on filter output')
        
            time, value = times['all'], values['all']

            close = np.abs(time - s1loc) < matchdist
            s1value = value[close]
        
            x = np.concatenate([s1value, s1value[-1:]])
            y = np.arange(len(x))[::-1] / nmc
            ax.plot(x, y, drawstyle='steps-pre', **plotkw['all'])

            textbox.textbox(ax, info)
        
            ax.minorticks_on()
            ax.grid(True, which='major', linestyle='--')
            ax.grid(True, which='minor', linestyle=':')
            ax.set_ylim(0, max(1, np.max(y)))
    
        
            figname = 'temps1.simulation_' + fname.replace(" ", "_") + '_combined'
            fig, ax = plt.subplots(num=figname, clear=True)
            figs.append(fig)
            ax.set_title(f'{fname.capitalize()} filter detection performance')
        
            ax.set_xlabel('S1 candidates per event')
            ax.set_ylabel('S1 loss probability')
        
            time, value = times['all'], values['all']
        
            close = np.abs(time - s1loc) < matchdist
            xs1 = value[close]
            ys1 = 1 - (1 + np.arange(len(xs1)))[::-1] / nmc
            fs1 = interpolate.interp1d(xs1, ys1, fill_value=(1 - ys1[0], 1), **interpkw)
        
            x, y = xy['all']
            sel = (xs1[0] <= x) & (x <= xs1[-1])
            x = x[sel]
            y = y[sel]
        
            s1cand = y
            s1prob = fs1(x)
        
            ax.plot(s1cand, s1prob, **plotkw['all'])
            textbox.textbox(ax, info)

            ax.minorticks_on()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, which='major', linestyle='--')
            ax.grid(True, which='minor', linestyle=':')
            l, r = ax.get_xlim()
            ax.set_xlim(max(0.1, l), r)
            b, _ = ax.get_ylim()
            ax.set_ylim(b, max(1, np.max(s1prob)))
            autolinscale(ax)

            figname = 'temps1.simulation_' + fname.replace(" ", "_") + '_time'
            fig, axs = plt.subplots(2, 1, num=figname, clear=True, figsize=[6.4, 7.19])
            figs.append(fig)
            axs[0].set_title(f'{fname.capitalize()} filter\nTemporal distribution of S1 candidates')
        
            ax = axs[0]
            ax.set_xlabel('Time relative to true S1 location [ns]')
            ax.set_ylabel('Inverse of neighbor temporal gap [ns$^{-1}$]')
        
            for k, time in times.items():
                time = np.sort(time)
                ddecdf = 1 / np.diff(time)
                x = time - s1loc
                y = np.concatenate([ddecdf, ddecdf[-1:]])
                ax.plot(x, y, drawstyle='steps-post', **plotkw[k])
        
            ax.axvspan(-deadradius, deadradius, color='#eee', zorder=-9, label='$\\pm$ dead radius')
            ax.axvspan(-matchdist,  matchdist,  color='#ccc', zorder=-8, label='$\\pm$ match dist.')

            ax.legend(loc='upper right')
            # ax.set_xscale('symlog', linthreshx=deadradius, linscalex=2)
            ax.set_xlim(3.5 * max(2 * matchdist, deadradius) * np.array([-1, 1]))
            ax.set_yscale('log')
            ax.minorticks_on()
            ax.grid(True, which='major', linestyle='--')
            ax.grid(True, which='minor', linestyle=':')
        
        
            ax = axs[1]
            ax.set_xlabel('Time relative to true S1 location [ns]')
            ax.set_ylabel('Histogram bin density [ns$^{-1}$]')
        
            times1 = hits1.reshape(-1)                
            time = times['all'] - s1loc
            time_match = time[np.abs(time) < matchdist]
            idx = np.argsort(np.abs(time))
            time_close = time[idx][:nmc]
        
            # t = np.linspace(left, right, 1000)
            # ax.plot(t, pS1.p_S1_gauss(t, VL, tauV, tauL, tres), label='S1 pdf')
            histkw = dict(bins='auto', density=True, histtype='step', zorder=10)
            ax.hist(times1, label=f'S1 photons ({len(times1)})', linestyle=':', **histkw)
            ax.hist(time_close, label=f'{nmc} closest candidates ($\\sigma_q$={qsigma.qsigma(time_close):.3g})', linestyle='--', **histkw)
            ax.hist(time_match, label=f'matching candidates ($\\sigma_q$={qsigma.qsigma(time_match):.3g})', **histkw)
        
            ax.axvspan(0, deadradius, color='#eee', zorder=-9, label='dead radius')
            ax.axvspan(0,  matchdist,  color='#ccc', zorder=-8, label='match dist.')

            textbox.textbox(ax, info, loc='upper left', zorder=11)
        
            ax.legend(loc='upper right', fontsize='small')
            ax.set_yscale('log')
            linthreshx = 10 ** np.ceil(np.log10(15 * qsigma.qsigma(time_match)))
            ax.set_xscale('symlog', linthreshx=linthreshx)
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(symloglocator.MinorSymLogLocator(linthreshx))
            ax.grid(True, which='major', linestyle='--')
            ax.grid(True, which='minor', linestyle=':')

        for fig in figs:
            fig.tight_layout()
            fig.show()
    
        return figs

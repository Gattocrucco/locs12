import numpy as np
from matplotlib import pyplot as plt

import pS1
import clusterargsort
import filters
import dcr
import textbox

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

def simulation(
    DCR=250e-9,       # (ns^-1) Dark count rate per PDM, 25 or 250 Hz
    VL=3,             # fast/slow ratio, ER=0.3, NR=3
    tauV=7,           # (ns) fast component tau
    tauL=1600,        # (ns) slow component tau
    T=4e6,            # (ns) time window (4 ms)
    npdm=8280,        # number of PDMs
    nphotons=10,      # (2-100) number of photons in the S1 signal
    tres=3,           # (ns) temporal resolution (3-10)
    nmc=100,          # number of simulated events
    deadradius=4000,  # (ns) for selecting S1 candidates in filter output
    matchdist=2000,   # (ns) for matching a S1 candidate to the true S1
    seed=202012191535 # random generator seed
):
    assert 2 * matchdist <= deadradius
    
    generator = np.random.default_rng(seed)

    hits1 = pS1.gen_S1((nmc, nphotons), VL, tauV, tauL, tres, generator)
    hitdcr = dcr.gen_DCR(nmc, T, DCR * npdm, generator)

    s1loc = T / 2
    hitall = np.concatenate([hits1 + s1loc, hitdcr], axis=-1)
    
    hitd = dict(all=hitall, dcr=hitdcr)
    plotkw = {
        'all': dict(label='Single S1 events'),
        'dcr': dict(label='No S1 events', linestyle='--')
    }

    filt = {
        k: filters.filters(hits, VL, tauV, tauL, tres, midpoints=1, pbar_batch=10)
        for k, hits in hitd.items()
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
        axs[0].set_title(f'S1 localization with temporal information\n{fname.capitalize()} filter')
        
        ax = axs[0]
        ax.set_ylabel('Mean number of S1 candidates per event')
        
        for k, value in values.items():            
            x = np.concatenate([value, value[-1:]])
            y = np.arange(len(x))[::-1] / nmc
            ax.plot(x, y, drawstyle='steps-pre', **plotkw[k])
        
        info = f"""\
DCR = {DCR * npdm * 1e3:.2g} $\\mu$s$^{{-1}}$
T = {T * 1e-6:.1f} ms
fast/slow = {VL:.1f}
nphotons = {nphotons}
$\\tau$ = ({tauV:.1f}, {tauL:.0f}) ns
temporal res. = {tres:.1f} ns
dead radius = {deadradius:.0f} ns
match dist. = {matchdist:.0f} ns"""

        textbox.textbox(ax, info)
        
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

        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
        ax.set_ylim(0, max(1, np.max(y)))

#         x = np.sort(fall[fname]['maxtime']) - s1loc
#         axh.plot(np.concatenate([x, x[-1:]]), np.linspace(0, 1, len(x) + 1), drawstyle='steps-pre')
#         axh.axvline(-s1tol, color='k')
#         axh.axvline(s1tol, color='k')
#
#         axh.set_xscale('symlog', linthreshx=10, linscalex=2)
#         axh.minorticks_on()
#         axh.grid(True, which='major', linestyle='--')
#         axh.grid(True, which='minor', linestyle=':')
#         axh.set_ylim(0, 1)
        
        fig.tight_layout()
        figs.append(fig)
    
        
        figname = 'temps1.simulation_' + fname.replace(" ", "_") + '_combined'
        fig, ax = plt.subplots(num=figname, clear=True)
        ax.set_title(f'S1 localization with temporal information\n{fname.capitalize()} filter')
        
        ax.set_xlabel('S1 candidates per event')
        ax.set_ylabel('S1 loss probability')
        
        time, value = times['all'], values['all']
        
        close = np.abs(time - s1loc) < matchdist
        closeidx = np.flatnonzero(close)
        repeat = np.concatenate([[1], np.diff(closeidx), [1]])
        
        s1cand = np.arange(1 + len(value))[::-1] / nmc
        s1prob = np.arange(1 + len(closeidx)) / nmc
        
        s1cand = s1cand[closeidx[0]:closeidx[-1] + 2]
        s1prob = np.repeat(s1prob, repeat)
        
        ax.plot(s1cand, s1prob, **plotkw['all'])
        textbox.textbox(ax, info)

        ax.minorticks_on()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
        ax.set_ylim(min(0, np.min(s1prob)), max(1, np.max(s1prob)))

        fig.tight_layout()
        figs.append(fig)

    for fig in figs:
        fig.show()
    
    return figs

import numpy as np
from matplotlib import pyplot as plt

import pS1
import clusterargsort
import filters
import dcr
import textbox

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
        
        sortedidx = {
            k: clusterargsort.clusterargsort(fhits[fname]['value'], fhits[fname]['time'], deadradius)
            for k, fhits in filt.items()
        }
        
        figname = 'temps1.simulation_' + fname.replace(" ", "_")
        fig, axs = plt.subplots(2, 1, num=figname, figsize=[6.4, 7.19], clear=True, sharex=True)
        axs[0].set_title(f'S1 localization with temporal information\n{fname.capitalize()} filter')
        
        ax = axs[0]
        ax.set_ylabel('Mean number of S1 candidates per event')
        
        for k, (indices1, length) in sortedidx.items():
            indices0 = np.repeat(np.arange(nmc), np.diff(length))
            values = filt[k][fname]['value'][indices0, indices1]
            values = np.sort(values)
            
            x = np.concatenate([values, values[-1:]])
            y = np.arange(len(x))[::-1] / nmc
            ax.plot(x, y, drawstyle='steps-pre', **plotkw[k])
        
        info = f"""\
DCR = {DCR * npdm * 1e3:.2g} $\\mu$s$^{{-1}}$
T = {T * 1e-6:.1f} ms
fast/slow = {VL:.1f}
nphotons = {nphotons}
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
        
        time = filt['all'][fname]['time']
        value = filt['all'][fname]['value']
        
        indices1, length = sortedidx['all']
        indices0 = np.repeat(np.arange(nmc), np.diff(length))
        
        ftime = time[indices0, indices1]
        fvalue = value[indices0, indices1]

        close = np.abs(ftime - s1loc) < matchdist
        
        s1value = np.sort(fvalue[close])
        
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
    
    for fig in figs:
        fig.show()
    
    return figs

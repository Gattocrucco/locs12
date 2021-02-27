import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

import textbox

rate = 10

table = np.load('temps1series0227.npy')

fig, ax = plt.subplots(num='temps1series0227plot', clear=True, figsize=[7.78, 5.41])

ax.set_title('S1 efficiency at fixed fake rate')
ax.set_xlabel('Number of S1 p.e.')
ax.set_ylabel(f'Efficiency at fake rate {rate} cps')

# shape is (DCR, VL, nphotons)
for ii, i in enumerate(np.ndindex(*table.shape[:-1])):
    entries = table[i]
    entries = entries[entries['done']]
    if len(entries) == 0:
        continue
    
    for ifilt, filt in enumerate(entries['filters'].dtype.names): 
        params = entries[0]['parameters']

        label = f'DCR={params["DCR"] * 1e9:.3g} cps/pdm, '
        label += 'ER' if params['VL'] < 1 else 'NR'
        if filt == 'cross correlation':
            label += f', crosscorr{params["sigma"]:.0f}'
        else:
            label += ', ' + filt

        nphotons = entries['parameters']['nphotons']
        interpkw = dict(kind='linear', assume_sorted=True, copy=False)
        efficiency = np.empty(len(entries))
        for ieff, entry in enumerate(entries):
            fentry = entry['filters'][filt]
            interp = interpolate.interp1d(entry['rate'], fentry['efficiencynointerp'], **interpkw)
            efficiency[ieff] = np.mean(interp(rate * (1 + 0.01 * np.array([-1, 1]))))
    
        color = ['#000', '#f55'][i[0]]
        plotkw = {
            'label': label,
            'color': color,
            'zorder': 2 - 0.001 * i[0],
            'linestyle': ['-', '--'][i[0]],
            'markerfacecolor': ['#fff', color][i[1]],
            'marker': ['v', '^', '.', 'o'][ifilt],
        }
        ax.plot(nphotons, efficiency, **plotkw)

ax.legend(loc='best', fontsize='small')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

def trimmedplot(ax, x, y, **kw):
    """
    Plot x vs y removing flat regions at the beginning and at the end.
    """
    changepoint = np.flatnonzero(np.diff(y))
    if len(changepoint) == 0:
        return ax.axhline(y[0], **kw)
    else:
        start = max(0     , changepoint[ 0] - 1)
        end   = min(len(y), changepoint[-1] + 3)
        sel = slice(start, end)
        return ax.plot(x[sel], y[sel], **kw)

def singleplot(idcr, ivl, inphotons, filter):
    fig, axs = plt.subplots(3, 1, num='temps1series0227.singleplot', clear=True, figsize=[6.4, 7.19])

    axs[0].set_title('Fake rate vs. threshold')
    axs[1].set_title('Efficiency vs. threshold')
    axs[2].set_title('Efficiency vs. fake rate')
    
    entry = table[idcr, ivl, inphotons]
    params = entry['parameters']
    info = f"""\
DCR={params["DCR"] * 1e9:.3g} cps/pdm
S1 {'ER' if params['VL'] < 1 else 'NR'} {params["nphotons"]} p.e.
{filter}"""
    if filter == 'cross correlation':
        info += f', $\\sigma$ = {params["sigma"]} ns'

    textbox.textbox(axs[0], info, fontsize='medium', loc='upper right')
    
    filt = entry['filters'][filter]
    
    trimmedplot(axs[0], entry['threshold'], filt['ratethr'])
    axs[0].axhline(rate, color='black', linestyle=':')
    axs[0].set_yscale('log')
    
    trimmedplot(axs[1], entry['threshold'], filt['effthr'])
    
    trimmedplot(axs[2], entry['rate'], filt['efficiency'], label='interp.', color='#f55')
    trimmedplot(axs[2], entry['rate'], filt['efficiencynointerp'], label='nointerp.', color='#000', linestyle='--')
    axs[2].axvline(rate, color='black', linestyle=':')
    axs[2].set_xscale('log')
    axs[2].legend()
    
    for ax in axs:
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
    
    fig.tight_layout()
    fig.show()
    return fig

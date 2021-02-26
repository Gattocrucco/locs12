import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

import textbox

rate = 10

table = np.load('temps1series02240-2.npy')

fig, ax = plt.subplots(num='temps1series02240plot', clear=True, figsize=[6.69, 4.66])

ax.set_title('S1 efficiency at fixed fake rate')
ax.set_xlabel('Number of S1 p.e.')
ax.set_ylabel(f'Efficiency at fake rate {rate} cps')

# shape is (DCR, VL, nphotons)
for ii, i in enumerate(np.ndindex(*table.shape[:-1])):
    entries = table[i]
    entries = entries[entries['done']]
    if len(entries) == 0:
        continue
    
    for ifilt, filt in enumerate(entries.dtype.names[-4:]):
        params = entries[0]['parameters']
    
        label = f'DCR={params["DCR"] * 1e9:.3g} cps/pdm, '
        label += 'ER' if params['VL'] < 1 else 'NR'
        label += ', ' + filt
    
        nphotons = entries['parameters']['nphotons']
        interpkw = dict(kind='linear', assume_sorted=True, copy=False)
        key = 'efficiencynointerp'
        # key = 'efficiency'
        # if filt.startswith('coinc'):
        #     key += 'nointerp'
        efficiency = [
            interpolate.interp1d(entry['rate'], entry[filt][key], **interpkw)(rate)
            for entry in entries
        ]
    
        # color = f'C{ii*3+ifilt}'
        color = ['#000', '#f55'][i[0]]
        plotkw = {
            'color': color,
            'zorder': 5 - i[0],
            'linestyle': ['-', '--'][i[0]],
            'markerfacecolor': ['#fff', color][i[1]],
            'marker': ['v', '^', '.', 'o'][ifilt],
        }
        ax.plot(nphotons, efficiency, label=label, **plotkw)

ax.legend(fontsize='small')
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
    fig, axs = plt.subplots(3, 1, num='temps1series02240plot.singleplot', clear=True, figsize=[6.4, 7.19])

    axs[0].set_title('Fake rate vs. threshold')
    axs[1].set_title('Efficiency vs. threshold')
    axs[2].set_title('Efficiency vs. fake rate')
    
    entry = table[idcr, ivl, inphotons]
    params = entry['parameters']
    info = f"""\
DCR={params["DCR"] * 1e9:.3g} cps/pdm
S1 {'ER' if params['VL'] < 1 else 'NR'} {params["nphotons"]} p.e.
{filter} filter"""

    textbox.textbox(axs[0], info, fontsize='medium', loc='upper right')
    
    filt = entry[filter]
    
    trimmedplot(axs[0], entry['threshold'], filt['ratethr'])
    axs[0].axhline(rate, color='black', linestyle='--')
    axs[0].set_yscale('log')
    
    trimmedplot(axs[1], entry['threshold'], filt['effthr'])
    
    trimmedplot(axs[2], entry['rate'], filt['efficiency'], label='interp.')
    trimmedplot(axs[2], entry['rate'], filt['efficiencynointerp'], label='nointerp.')
    axs[2].axvline(rate, color='black', linestyle='--')
    axs[2].set_xscale('log')
    axs[2].legend()
    
    for ax in axs:
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
    
    fig.tight_layout()
    fig.show()
    return fig

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate, optimize

import textbox

rate = 100
fixedeff = 0.5

table = np.load('temps1series0226z-2.npy')

fig, ax = plt.subplots(num='temps1series0226zplot', clear=True, figsize=[7.78, 5.41])

ax.set_title('S1 efficiency at fixed fake rate')
ax.set_xlabel('Number of S1 p.e.')
ax.set_ylabel(f'Efficiency at fake rate {rate} cps')

fixedeffnph = np.full(table.shape[:-1], np.nan)

# shape is (DCR, VL, sigma, nphotons)
for ii, i in enumerate(np.ndindex(*table.shape[:-1])):
    entries = table[i]
    entries = entries[entries['done']]
    if len(entries) == 0:
        continue
    
    params = entries[0]['parameters']

    label = f'DCR={params["DCR"] * 1e9:.3g} cps/pdm, '
    label += 'ER' if params['VL'] < 1 else 'NR'

    nphotons = entries['parameters']['nphotons']
    interpkw = dict(kind='linear', assume_sorted=True, copy=False)
    efficiency = np.empty(len(entries))
    for ieff, entry in enumerate(entries):
        fentry = entry['filters']['cross correlation']
        interp = interpolate.interp1d(entry['rate'], fentry['efficiencynointerp'], **interpkw)
        efficiency[ieff] = np.mean(interp(rate * (1 + 0.01 * np.array([-1, 1]))))
    
    try:
        # interpkw.update(kind='quadratic')
        feff = interpolate.interp1d(nphotons, efficiency, **interpkw)
        result = optimize.root_scalar(lambda x: feff(x) - fixedeff, bracket=(np.min(nphotons), np.max(nphotons)))
        assert result.converged, result.flag
        fixedeffnph[i] = result.root
    except ValueError:
        pass

    color = ['#000', '#f55'][i[0]]
    plotkw = {
        'color': color,
        'alpha': (1 + i[2]) / table.shape[2],
        'zorder': 2 - 0.001 * i[0],
        'linestyle': ['-', '--'][i[0]],
        'marker': ['o', '^'][i[1]],
        'markerfacecolor': ['#fff', color][i[1]],
    }
    if i[0] == i[1] == 0:
        label += f', sigma={params["sigma"]}ns'
        plotkw['label'] = label
    elif i[2] == table.shape[2] - 1:
        plotkw['label'] = label
    ax.plot(nphotons, efficiency, **plotkw)

ax.legend(loc='best', fontsize='small')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

fig, ax = plt.subplots(num='temps1series0226zplot-2', clear=True)

ax.set_title('S1 p.e. at fixed efficiency and fake rate')
ax.set_xlabel('Total $\\sigma$ of cross correlation template [ns]')
ax.set_ylabel(f'S1 p.e. with {fixedeff*100:.0f} % efficiency at fake rate {rate} cps')

print(f'fake rate = {rate} cps, efficiency = {fixedeff * 100:.0f} %')

# shape is (DCR, VL, sigma, nphotons)
for ii, i in enumerate(np.ndindex(table.shape[:-2])):
    nphotons = fixedeffnph[i]
    entries = table[i]
    if np.count_nonzero(entries['done']) == 0:
        continue
    
    params = entries[0, 0]['parameters']

    label = f'DCR={params["DCR"] * 1e9:.3g} cps/pdm, '
    label += 'ER' if params['VL'] < 1 else 'NR'

    sigma = entries[:, 0]['parameters']['sigma']

    color = ['#000', '#f55'][i[0]]
    plotkw = {
        'color': color,
        'zorder': 10 - i[0],
        'linestyle': ['-', '--'][i[0]],
        'marker': ['o', '^'][i[1]],
        'markerfacecolor': ['#fff', color][i[1]],
        'label': label
    }
    ax.plot(sigma, nphotons, **plotkw)
    
    idx = np.argmin(nphotons)
    print(f'{label}: sigma for min photons = {sigma[idx]}')

ax.legend(loc='best')
ax.set_xscale('log')
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

def singleplot(idcr, ivl, isigma, inphotons):
    fig, axs = plt.subplots(3, 1, num='temps1series0226zplot.singleplot', clear=True, figsize=[6.4, 7.19])

    axs[0].set_title('Fake rate vs. threshold')
    axs[1].set_title('Efficiency vs. threshold')
    axs[2].set_title('Efficiency vs. fake rate')
    
    entry = table[idcr, ivl, isigma, inphotons]
    params = entry['parameters']
    info = f"""\
DCR={params["DCR"] * 1e9:.3g} cps/pdm
S1 {'ER' if params['VL'] < 1 else 'NR'} {params["nphotons"]} p.e.
cross correlation, $\\sigma$ = {params["sigma"]} ns"""

    textbox.textbox(axs[0], info, fontsize='medium', loc='upper right')
    
    filt = entry['filters']['cross correlation']
    
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

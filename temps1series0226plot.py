import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate, optimize

rate = 10
fixedeff = 0.5

table = np.load('temps1series0226.npy')

fig, ax = plt.subplots(num='temps1series0226plot', clear=True, figsize=[7.78, 5.41])

ax.set_title('S1 efficiency at fixed fake rate')
ax.set_xlabel('Number of S1 p.e.')
ax.set_ylabel(f'Efficiency at fake rate {rate} cps')

fixedeffnph = np.full(table.shape[:-1] + (len(table['filters'].dtype.names),), np.nan)

# shape is (DCR, VL, nphotons)
for i in np.ndindex(*table.shape[:-1]):
    entries = table[i]
    entries = entries[entries['done']]
    if len(entries) == 0:
        continue
    
    params = entries[0]['parameters']

    label = f'DCR={params["DCR"] * 1e9:.3g} cps/pdm, '
    label += 'ER' if params['VL'] < 1 else 'NR'

    nphotons = entries['parameters']['nphotons']
    interpkw = dict(kind='linear', assume_sorted=True, copy=False)
    
    filters = entries['filters'].dtype.names
    for ifname, fname in enumerate(filters):
        tcoinc = float(fname[5:])
        
        efficiency = [
            interpolate.interp1d(entry['rate'], entry['filters'][fname]['efficiencynointerp'], **interpkw)(rate)
            for entry in entries
        ]
    
        feff = interpolate.interp1d(nphotons, efficiency, **interpkw)
        try:
            result = optimize.root_scalar(lambda x: feff(x) - fixedeff, bracket=(np.min(nphotons), np.max(nphotons)))
            assert result.converged, result.flag
            fixedeffnph[i + (ifname,)] = result.root
        except ValueError:
            pass

        color = ['#000', '#f55'][i[0]]
        plotkw = {
            'color': color,
            'alpha': (1 + ifname) / len(filters),
            'zorder': 2 - 0.001 * i[0],
            'linestyle': ['-', '--'][i[0]],
            'marker': ['o', '^'][i[1]],
            'markerfacecolor': ['#fff', color][i[1]],
        }
        if i[0] == i[1] == 0:
            plotkw['label'] = label + f', T={tcoinc:.0f}ns'
        elif ifname == len(filters) - 1:
            plotkw['label'] = label
        ax.plot(nphotons, efficiency, **plotkw)

ax.legend(loc='best', fontsize='small')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

fig, ax = plt.subplots(num='temps1series0226plot-2', clear=True)

ax.set_title('S1 p.e. at fixed efficiency and fake rate')
ax.set_xlabel('Duration of coincidence window [ns]')
ax.set_ylabel(f'S1 p.e. with {fixedeff*100:.0f} % efficiency at fake rate {rate} cps')

# shape is (DCR, VL, nphotons)
for i in np.ndindex(table.shape[:-1]):
    nphotons = fixedeffnph[i]
    entries = table[i]
    if np.count_nonzero(entries['done']) == 0:
        continue
    
    params = entries[0]['parameters']

    label = f'DCR={params["DCR"] * 1e9:.3g} cps/pdm, '
    label += 'ER' if params['VL'] < 1 else 'NR'

    tcoinc = [float(f[5:]) for f in entries['filters'].dtype.names]

    color = ['#000', '#f55'][i[0]]
    plotkw = {
        'color': color,
        'zorder': 2 - 0.001 * i[0],
        'linestyle': ['-', '--'][i[0]],
        'marker': ['o', '^'][i[1]],
        'markerfacecolor': ['#fff', color][i[1]],
        'label': label
    }
    ax.plot(tcoinc, nphotons, **plotkw)

ax.legend(loc='best')
ax.set_xscale('log')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

rate = 10

table = np.load('temps1series0222.npy')

fig, ax = plt.subplots(num='temps1series0222plot', clear=True)

ax.set_title('S1 efficiency at fixed fake rate')
ax.set_xlabel('Number of S1 p.e.')
ax.set_ylabel(f'Efficiency at fake rate {rate} cps')

# shape is (DCR, VL, nphotons)
for ii, i in enumerate(np.ndindex(*table.shape[:-1])):
    entries = table[i]
    entries = entries[entries['done']]
    if len(entries) == 0:
        continue
    
    for ifilt, filt in enumerate(entries.dtype.names[-3:]):
        params = entries[0]['parameters']
    
        label = f'DCR={params["DCR"] * 1e9:.3g} cps/pdm, '
        label += 'ER' if params['VL'] < 1 else 'NR'
        label += ', ' + filt
    
        nphotons = entries['parameters']['nphotons']
        interpkw = dict(kind='linear', assume_sorted=True, copy=False)
        efficiency = [
            interpolate.interp1d(entry['rate'], entry[filt]['efficiency'], **interpkw)(rate)
            for entry in entries
        ]
    
        color = f'C{ii*3+ifilt}'
        plotkw = {
            'color': color,
            'linestyle': ['-', '--'][i[0]],
            'markerfacecolor': ['#fff', color][i[1]],
            'marker': ['v', '^', '.'][ifilt],
        }
        ax.plot(nphotons, efficiency, label=label, **plotkw)

ax.legend(loc='best', fontsize='small')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

rate = 10

table = np.load('temps1series02240.npy')

fig, ax = plt.subplots(num='temps1series02240plot', clear=True, figsize=[10, 4.5])

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
        key = 'efficiency'
        if filt.startswith('coinc'):
            key += 'nointerp'
        efficiency = [
            interpolate.interp1d(entry['rate'], entry[filt][key], **interpkw)(rate)
            for entry in entries
        ]
    
        # color = f'C{ii*3+ifilt}'
        color = ['#000', '#f55'][i[0]]
        plotkw = {
            'color': color,
            'zorder': 10 - i[0],
            'linestyle': ['-', '--'][i[0]],
            'markerfacecolor': ['#fff', color][i[1]],
            'marker': ['v', '^', '.', 'o'][ifilt],
        }
        ax.plot(nphotons, efficiency, label=label, **plotkw)

ax.legend(fontsize='small')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')
ax.set_xlim(0, 50)

fig.tight_layout()
fig.show()

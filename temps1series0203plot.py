import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

rate = 10

table = np.load('temps1series0203.npy')

fig, ax = plt.subplots(num='temps1series0203plot', clear=True)

ax.set_title('S1 efficiency at fixed fake rate')
ax.set_xlabel('Number of S1 photons')
ax.set_ylabel(f'Efficiency at fake rate {rate} s$^{{-1}}$')

for ii, i in enumerate(np.ndindex(*table.shape[:-1])):
    entries = table[i]
    entries = entries[entries['done']]
    if len(entries) == 0:
        continue
    
    params = entries[0]['parameters']
    params['DCR'] *= 1e9
    units = dict(DCR=' Hz/pdm', tres=' ns')
    label = ', '.join([
        f'{name}={params[name]:.3g}{units.get(name, "")}'
        for name in params.dtype.names[:-1]
    ])
    
    nphotons = entries['parameters']['nphotons']
    interpkw = dict(kind='linear', assume_sorted=True, copy=False)
    efficiency = [
        interpolate.interp1d(entry['rate'], entry['efficiency'], **interpkw)(rate)
        for entry in entries
    ]
    
    color = f'C{ii}'
    plotkw = {
        'color': color,
        'linestyle': ['-', '--'][i[0]],
        'markerfacecolor': ['#fff', color][i[1]],
        'marker': ['v', '^'][i[2]],
    }
    ax.plot(nphotons, efficiency, label=label, **plotkw)

ax.legend(loc='best', fontsize='small')
ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

import textbox

inphotons = 2 # in [3, 7, 10, 15, 20, 30, 40]
idcr = 1 # in [25, 250] cps/pdm

table = np.load('temps1series0213.npy')

figkw = dict(clear=True, sharex=True, figsize=[9, 7])
figs = []
axs = []
for wname in ['efficiency', 'fakerate', 'effvsrate']:
    figkw['sharey'] = 'col' if wname == 'effvsrate' else True
    fig, ax = plt.subplots(2, 2, num=f'temps1series0213plot-{wname}', **figkw)
    figs.append(fig)
    axs.append(ax)
axs = np.array(axs)

for ax in axs[0].reshape(-1):
    if ax.is_first_col():
        ax.set_ylabel('S1 detection efficiency')
    if ax.is_last_row():
        ax.set_xlabel('Threshold on filter output')

for ax in axs[1].reshape(-1):
    if ax.is_first_col():
        ax.set_ylabel('Fake rate [cps]')
    if ax.is_last_row():
        ax.set_xlabel('Threshold on filter output')

for ax in axs[2].reshape(-1):
    if ax.is_first_col():
        ax.set_ylabel('S1 detection efficiency')
    if ax.is_last_row():
        ax.set_xlabel('Fake rate [cps]')

# the shape of table is over (DCR, VL, nphotons, sigma)

for ivl in range(table.shape[1]):
    entries = table[idcr, ivl, inphotons]
    if np.count_nonzero(entries['done']) == 0:
        continue
    
    for ifilter, fname in enumerate(['ER', 'NR']):
        qax = axs[:, ifilter, ivl]
        
        for isigma, entry in enumerate(entries):
            if not entry['done']:
                continue
            
            sigma = entry['parameters']['sigma']
            plotkw = dict(
                alpha=(isigma + 1) / len(entries),
                color='#600',
                label=f'{sigma:.3g}',
                linestyle=['-', '--', '-.', ':'][isigma % 3]
            )
            
            for ifig, ax in enumerate(qax):
                
                if ifig == 0:
                    x = entry['threshold']
                    y = entry[fname]['effthr']
                elif ifig == 1:
                    x = entry['threshold']
                    y = entry[fname]['ratethr']
                elif ifig == 2:
                    x = entry['rate']
                    y = entry[fname]['efficiency']
                
                changepoint = np.flatnonzero(np.diff(y))
                start = max(0     , changepoint[ 0] - 1)
                end   = min(len(y), changepoint[-1] + 3)
                sel = slice(start, end)
                x = x[sel]
                y = y[sel]
                
                ax.plot(x, y, **plotkw)
        
        for ax in qax:
            s1type = 'ER' if entries[0]['parameters']['VL'] < 1 else 'NR'
            ax.set_title(f'{s1type} S1, {fname} filter')
            ax.minorticks_on()
            ax.grid(True, which='major', linestyle='--')
            ax.grid(True, which='minor', linestyle=':')

for ifig, fax in enumerate(axs):
    fax[0, 0].legend(loc='best', fontsize='small', ncol=2, title='Template $\\sigma$ [ns]')

    params = table[idcr, 0, inphotons, 0]['parameters']
    info = f"""\
    DCR = {params['DCR'] * 1e9:.3g} cps/pdm
    nphotons = {params['nphotons']}
    tres = 10 ns
    nevents = 1000"""
    
    infoheight = 'lower' if ifig == 2 else 'upper'
    textbox.textbox(fax[0, 1], info, loc=f'{infoheight} right', fontsize='small')
    
    if ifig == 1:
        fax[0, 0].set_yscale('log')
    if ifig == 2:
        fax[0, 0].set_xscale('log')

for fig in figs:
    fig.tight_layout()
    fig.show()

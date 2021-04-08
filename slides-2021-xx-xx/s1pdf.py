from matplotlib import pyplot as plt
import numpy as np

import ps12
import textbox

tau1 = 7
tau2 = 1600
sigma = 10

fig, ax = plt.subplots(num='slides-2021-02-16.s1pdf', clear=True)

ax.set_xlabel('Time [ns]')
ax.set_ylabel('Probability density [ns$^{-1}$]')

t = np.linspace(-5 * sigma, tau2 / 2, 1000)
s1er = ps12.ps1gauss(t, 0.25, tau1, tau2, sigma)
s1nr = ps12.ps1gauss(t, 0.75, tau1, tau2, sigma)

ax.plot(t, s1nr, label='NR', color='#f55')
ax.plot(t, s1er, label='ER', color='black')

info = f"""\
$\\tau_{{\\mathrm{{fast}}}}$ = {tau1} ns
$\\tau_{{\\mathrm{{slow}}}}$ = {tau2} ns
$\\sigma$ = {sigma} ns"""
textbox.textbox(ax, info, fontsize='medium', loc='upper right')

ax.legend(title='S1', loc='upper center')

ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

from matplotlib import pyplot as plt
import numpy as np

import ps12
import textbox
import dcr

n = 10
p1 = 0.25
tau1 = 7
tau2 = 1600
sigma = 10
loc = 100

T = 2000
rate = 0.005
locs = [loc, loc + 1000]

gen = np.random.default_rng(202102151206)

fig, ax = plt.subplots(num='slides-2021-02-16.ccexample', clear=True)

ax.set_xlabel('Time [ns]')

s1hit = loc + ps12.gens12(n, p1, tau1, tau2, 0, sigma, gen)
s1hit = s1hit[s1hit < T]
dchit = dcr.gen_DCR((), T, rate, gen)
hitkw = [
    (s1hit, dict(color='#f55', label='S1 photons'        )),
    (dchit, dict(color='#000', label='dark count photons')),
]

for hit, kw in hitkw:
    for h in hit:
        ax.axvline(h, 0, 0.5, **kw)
        kw.pop('label', None)

t = np.arange(0, T)
hits = np.concatenate([s1hit, dchit])
params = (p1, tau1, tau2, sigma)
for x in locs:
    s1 = ps12.ps1gauss(t - x, *params)
    ax.fill_between(t, np.zeros_like(s1), s1, label='$p_{\\mathrm{S1}}' + f'(t - {x})$', alpha=0.7)
    sel = hits > x - 3 * sigma
    ax.plot(hits[sel], ps12.ps1gauss(hits[sel] - x, *params), 'xk')

info = f"""\
$p_{{\\mathrm{{fast}}}}$ = {p1}
$\\tau_{{\\mathrm{{fast}}}}$ = {tau1} ns
$\\tau_{{\\mathrm{{slow}}}}$ = {tau2} ns
$\\sigma$ = {sigma} ns
DCR = {rate * 1000} $\\mu$s$^{{-1}}$"""
textbox.textbox(ax, info, fontsize='medium', loc='center right', bbox=dict(alpha=0.95))

ax.legend(loc='upper right')

ax.minorticks_on()
ax.grid(True, which='major', linestyle='--')
ax.grid(True, which='minor', linestyle=':')

fig.tight_layout()
fig.show()

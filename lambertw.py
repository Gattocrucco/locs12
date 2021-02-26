from matplotlib import pyplot as plt
from scipy import special, integrate
import numpy as np
# import numba
#
# # @numba.cfunc('f8(f8,f8,f8)')
# def integrand(u, q, T):
#     return u ** q * np.exp(-T * (1 - u))
#
# @np.vectorize
# def integral(q, T):
#     result = integrate.quad(integrand, np.exp(-T), 1, args=(q, T), points=(1,))
#     return result[0]
#
# def equation(q, T):
#     return q + np.expm1(T * np.expm1(-T)) + T * integral(q, T)
#
# fig, ax = plt.subplots(num='lambertw2', clear=True)
#
# q = np.linspace(0.01, 1, 100)
# mu = np.logspace(np.log10(0.1), np.log10(10), 10)
# for i, T in enumerate(mu):
#     ax.plot(q, integral(q, T), label=f'T={T:.2g}', color='#000', alpha=(1 + i) / len(mu))
#
# ax.legend(fontsize='small')
# ax.minorticks_on()
# ax.grid(True, 'major', linestyle='--')
# ax.grid(True, 'minor', linestyle=':')
#
# fig.tight_layout()
# fig.show()

def f(x):
    return special.exprel(-x)

def g(x):
    z = special.lambertw(x) / x
    imz = np.imag(z)
    assert np.all((imz == 0) | np.isnan(imz))
    return np.where(x == 0, 1, np.real(z))

def h(x):
    return 1/(1+x)


functions = [f, g, h]
formulas = [
    'exprel(-x)',
    'W(x)/x',
    '1/(1+x)',
]

fig, axs = plt.subplots(3, 1, num='lambertw', clear=True, figsize=[6.4, 7.19])

for fun, label in zip(functions, formulas):
    x = np.linspace(0, 10, 1000)
    axs[0].plot(x, fun(x), label=label)
    
    x = np.logspace(0, 3, 1000)
    axs[1].plot(x, fun(x) * x, label=f'({label}) / (1/x)')
    
    x = np.logspace(-3, 0, 1000)
    axs[2].plot(x, (1 - fun(x)) / x, label=f'(1 - {label}) / x')

for iax, ax in enumerate(axs):
    if iax != 0:
        ax.axhline(1, linestyle='--', color='k', zorder=2)
        ax.set_xscale('log')
    ax.legend()
    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')

fig.tight_layout()
fig.show()

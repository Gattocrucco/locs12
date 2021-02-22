"""
Module to compute and sample the temporal distribution of S1 photons.
"""

import numpy as np
from scipy import stats, special, optimize
from matplotlib import pyplot as plt
import numba

import numba_scipy_special
import textbox

def p_exp(t, tau):
    """
    Exponential pdf with scale tau.
    """
    return stats.expon.pdf(t, scale=tau)

def p_gauss(t, sigma):
    """
    Gaussian pdf with scale sigma.
    """
    return stats.norm.pdf(t, scale=sigma)

def p_S1(t, VL, tauV, tauL):
    """
    Mixture of two exponential pdfs "V" and "L", VL is the ratio of the
    coefficients V to L, tauV and tauL are the scale parameters.
    """
    V = 1 / (1 + 1 / VL)
    L = 1 / (1 + VL)
    return V * p_exp(t, tauV) + L * p_exp(t, tauL)

@numba.vectorize(nopython=True)
def log_p_exp_gauss(t, tau, sigma):
    """
    (Logarithm of) exponential pdf with scale tau convoluted with a normal with
    scale sigma.
    """
    return -np.log(tau) - t/tau + 1/2 * (sigma / tau) ** 2 + special.log_ndtr(t / sigma - sigma / tau)

@numba.njit
def p_exp_gauss(t, tau, sigma):
    """
    Exponential pdf with scale tau convoluted with a normal with scale sigma.
    """
    return np.exp(log_p_exp_gauss(t, tau, sigma))

@numba.njit
def p_S1_gauss(t, VL, tauV, tauL, sigma):
    """
    p_S1 convoluted with a normal with scale sigma.
    """
    V = 1 / (1 + 1 / VL)
    L = 1 / (1 + VL)
    return V * p_exp_gauss(t, tauV, sigma) + L * p_exp_gauss(t, tauL, sigma)

@numba.njit
def log_p_S1_gauss(t, VL, tauV, tauL, sigma):
    """
    (Logarithm of) p_S1 convoluted with a normal with scale sigma.
    """
    V = 1 / (1 + 1 / VL)
    L = 1 / (1 + VL)
    return np.logaddexp(np.log(V) + log_p_exp_gauss(t, tauV, sigma), np.log(L) + log_p_exp_gauss(t, tauL, sigma))

@numba.njit
def log_likelihood(t, VL, tauV, tauL, sigma, dcr, nph):
    """
    Logarithm of the distribution of S1 + dcr.
    dcr = total dark count rate
    nph = number of S1 photons
    """
    return np.logaddexp(0, np.log(nph / dcr) + log_p_S1_gauss(t, VL, tauV, tauL, sigma))

def p_S1_gauss_maximum(VL, tauV, tauL, sigma):
    """
    Return the position of the maximum of p_S1_gauss with the given parameters.
    """
    fun = lambda t: -log_p_S1_gauss(t, VL, tauV, tauL, sigma)
    bracket = (0, sigma)
    result = optimize.minimize_scalar(fun, bracket)
    assert result.success
    assert -sigma <= result.x <= 5 * sigma
    return result.x

def p_exp_gauss_maximum(tau, sigma):
    """
    Return the position of the maximum of p_exp_gauss with the given parameters.
    """
    fun = lambda t: -log_p_exp_gauss(t, tau, sigma)
    bracket = (0, sigma)
    result = optimize.minimize_scalar(fun, bracket)
    assert result.success
    assert -sigma <= result.x <= 5 * sigma
    return result.x

def check_p_S1(VL=3, tauV=7, tauL=1600, tres=10):
    """
    plot p_S1 and p_S1_gauss.
    """
    fig, ax = plt.subplots(num='pS1.check_p_S1', clear=True)
    
    ax.set_title('Time distribution of S1 photons')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Probability density [ns$^{-1}$]')
    
    t = np.linspace(-3 * tres, tauL, 10000)
    y = p_S1(t, VL, tauV, tauL)
    yc = p_S1_gauss(t, VL, tauV, tauL, tres)
    yc2 = np.exp(log_p_S1_gauss(t, VL, tauV, tauL, tres))
    yg = p_gauss(t, tres)
    
    mx = p_S1_gauss_maximum(VL, tauV, tauL, tres)
    my = p_S1_gauss(mx, VL, tauV, tauL, tres)
    
    ax.plot(t, y, label='S1')
    ax.plot(t, yc, label='S1 + res')
    ax.plot(t, yc2, '--', label='S1 + res (explog)')
    ax.plot(t, yg, label='res')
    ax.plot(mx, my, 'xk', label='maximum')
    
    ax.legend(loc='best')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')
    ax.set_yscale('log')
    ax.set_ylim(np.min(yc), np.max(y))

    fig.tight_layout()
    fig.show()
    
    return fig

def check_likelihood(VL=3, tauV=7, tauL=1600, tres=10, dcr=2.5e-3, nph=10):
    """
    plot log_likelihood.
    """
    fig, ax = plt.subplots(num='pS1.check_likelihood', clear=True)
    
    ax.set_title('Likelihood of S1 + dark count')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Log likelihood')
    
    t = np.linspace(-5 * tres, tauL, 10000)
    y = log_likelihood(t, VL, tauV, tauL, tres, dcr, nph)
    y2 = p_S1_gauss(t, VL, tauV, tauL, tres)
    
    mx = p_S1_gauss_maximum(VL, tauV, tauL, tres)
    my = log_likelihood(mx, VL, tauV, tauL, tres, dcr, nph)
    
    ax.plot(t, y, label='likelihood')
    ax.plot(mx, my, 'x', label='maximum')
    ax.plot(t, y2 / np.max(y2) * np.max(y), label='p_S1_gauss (arb.un.)')
    
    ax.legend(loc='best')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')

    fig.tight_layout()
    fig.show()
    
    return fig

def gen_S1(size, VL, tauV, tauL, tres, generator=None):
    """
    Generate samples from the pdf p_S1_gauss.
    """
    size = (size,) if np.isscalar(size) else tuple(size)
    if generator is None:
        generator = np.random.default_rng()
    
    expsamp = generator.standard_exponential(size=size + (2,))
    expsamp *= [tauV, tauL]
    choice = generator.binomial(n=1, p=1 / (1 + VL), size=size)
    ogrid_indices = tuple(slice(s) for s in size)
    indices = tuple(np.ogrid[ogrid_indices])
    samp = expsamp[indices + (choice,)]
    assert samp.shape == size
    
    normsamp = generator.standard_normal(size=size)
    normsamp *= tres
    samp += normsamp
    
    return samp

def check_gen_S1(VL=3, tauV=10, tauL=100, tres=5):
    """
    Check gen_S1.
    """
    fig, ax = plt.subplots(num='pS1.check_gen_S1', clear=True)
    
    ax.set_title('Time distribution of S1 photons')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Probabilty density [ns$^{-1}$]')

    gen = np.random.default_rng(202012181733)
    s = gen_S1((100, 100, 10), VL, tauV, tauL, tres, gen).reshape(-1)
    left, right = np.quantile(s, [0.001, 0.999])
    s = s[(s >= left) & (s <= right)]
    
    ax.hist(s, bins='auto', histtype='step', density=True, label='samples histogram')
    
    t = np.linspace(np.min(s), np.max(s), 10000)
    y = p_S1_gauss(t, VL, tauV, tauL, tres)
    
    ax.plot(t, y, label='pdf')
    
    ax.legend(loc='best')
    ax.grid()
    ax.set_yscale('log')
    
    fig.tight_layout()
    fig.show()
    
    return fig

if __name__ == '__main__':
    check_p_S1()
    check_gen_S1()

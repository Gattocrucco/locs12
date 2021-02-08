"""
Module to compute and sample the temporal distribution of S1 and S2 photons.
"""

import numpy as np
from scipy import stats, special, optimize
from matplotlib import pyplot as plt
import numba

import numba_scipy_special
import sampling_bounds

def pexp(t, tau):
    """
    Exponential pdf with scale tau.
    """
    return stats.expon.pdf(t, scale=tau)

def pgauss(t, sigma):
    """
    Normal pdf with scale sigma.
    """
    return stats.norm.pdf(t, scale=sigma)

def ps1(t, p1, tau1, tau2):
    """
    Mixture of two exponential pdfs with scale parameters tau1 and tau2. p1
    is the amplitude of the tau1 component.
    """
    return p1 * pexp(t, tau1) + (1 - p1) * pexp(t, tau2)

@numba.vectorize(nopython=True)
def pexpunif(t, tau, T):
    """
    Convolution of an exponential with scale tau and a uniform in (0, T).
    """
    if t <= 0:
        return 0
    elif 0 < t <= T:
        return -np.expm1(-t / tau) / T
    else:
        return np.exp(-(t - T) / tau) * special.exprel(-T / tau) / tau

def ps2(t, p1, tau1, tau2, T):
    """
    Mixture of two exponentials with scales tau1 and tau2 convoluted with
    a uniform in (0, T). p1 is the weight of the tau1 component.
    """
    return p1 * pexpunif(t, tau1, T) + (1 - p1) * pexpunif(t, tau2, T)

@numba.vectorize(nopython=True)
def logq(t, tau, sigma):
    """
    Logarithm of exponential pdf with scale tau convoluted with a normal with
    scale sigma without the 1/tau factor.
    """
    return -t / tau + 1/2 * (sigma / tau) ** 2 + special.log_ndtr(t / sigma - sigma / tau)

@numba.njit
def logpexpgauss(t, tau, sigma):
    """
    Logarithm of exponential pdf with scale tau convoluted with a normal with
    scale sigma.
    """
    return logq(t, tau, sigma) - np.log(tau)

@numba.njit
def pexpgauss(t, tau, sigma):
    """
    Exponential pdf with scale tau convoluted with a normal with scale sigma.
    """
    return np.exp(logpexpgauss(t, tau, sigma))

@numba.vectorize(nopython=True)
def pexpunifgauss(t, tau, T, sigma):
    """
    Convolution of an exponential with scale tau, a uniform in (0, T), and a
    normal with scale sigma.
    """
    return 1/T * (special.ndtr(t / sigma) - special.ndtr((t - T) / sigma) - np.exp(logq(t, tau, sigma)) + np.exp(logq(t - T, tau, sigma)))

@numba.njit
def ps1gauss(t, p1, tau1, tau2, sigma):
    """
    ps1 convoluted with a normal with scale sigma.
    """
    return p1 * pexpgauss(t, tau1, sigma) + (1 - p1) * pexpgauss(t, tau2, sigma)

@numba.njit
def logps1gauss(t, p1, tau1, tau2, sigma):
    """
    Logarithm of ps1 convoluted with a normal with scale sigma.
    """
    return np.logaddexp(
        np.log(p1)    + logpexpgauss(t, tau1, sigma),
        np.log1p(-p1) + logpexpgauss(t, tau2, sigma)
    )

@numba.njit
def ps2gauss(t, p1, tau1, tau2, T, sigma):
    """
    ps2 convoluted with a normal with scale sigma.
    """
    return p1 * pexpunifgauss(t, tau1, T, sigma) + (1 - p1) * pexpunifgauss(t, tau2, T, sigma)

def ps1gaussmax(p1, tau1, tau2, sigma):
    """
    Return the position of the maximum of ps1gauss with the given parameters.
    """
    fun = lambda t: -logps1gauss(t, p1, tau1, tau2, sigma)
    bracket = (0, sigma)
    result = optimize.minimize_scalar(fun, bracket)
    assert result.success
    assert -sigma <= result.x <= 5 * sigma
    return result.x

def ps2gaussmax(p1, tau1, tau2, T, sigma):
    """
    Return the position of the maximum of ps2gauss with the given parameters.
    """
    fun = lambda t: -np.log(ps2gauss(t, p1, tau1, tau2, T, sigma))
    bracket = (0, T/2)
    result = optimize.minimize_scalar(fun, bracket)
    assert result.success
    assert -sigma <= result.x <= T + sigma
    return result.x

def ps12range(p1, tau1, tau2, T, sigma, nsamples=1, p=0.01):
    """
    Return an interval (left, right) that approximately will contain all the
    samples from ps2gauss with probability `p` if `nsamples` samples are drawn.
    """
    left, _ = sampling_bounds.sampling_bounds('norm', nsamples, p)
    left *= sigma
    
    neff = nsamples * (p1 if tau1 > tau2 else 1 - p1)
    _, right = sampling_bounds.sampling_bounds('expon', neff, p)
    right *= max(tau1, tau2)
    right += T
    
    return left, right

def gens12(size, p1, tau1, tau2, T, sigma, generator=None):
    """
    Draw samples from ps2gauss. In particular if T = 0 the distribution is
    ps1gauss.
    
    Parameters
    ----------
    size : int or tuple
        The length or shape of the samples array.
    p1, tau1, tau2, T, sigma : scalar
        The parameters of ps2gauss.
    generator : np.random.Generator, optional
        Random generator.
    
    Return
    ------
    samp : array
        Samples array with shape specified by `size`.
    """
    size = (size,) if np.isscalar(size) else tuple(size)
    if generator is None:
        generator = np.random.default_rng()
    
    samp = generator.standard_exponential(size=size)
    choice = generator.binomial(n=1, p=p1, size=size).astype(bool)
    samp[choice] *= tau1
    samp[~choice] *= tau2
    # alternative: generate # of tau1, tau2 with binomial on the total,
    # multiply by tau2, tau1, then use generator.shuffle()
    
    if sigma != 0:
        normsamp = generator.standard_normal(size=size)
        normsamp *= sigma
        samp += normsamp
    
    if T != 0:
        unifsamp = generator.uniform(0, T, size=size)
        samp += unifsamp
    
    return samp

def check_ps1(p1=0.75, tau1=7, tau2=1600, sigma=10):
    """
    Plot ps1 and ps1gauss.
    """
    fig, ax = plt.subplots(num='ps12.check_ps1', clear=True)
    
    ax.set_title('Time distribution of S1 photons')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Probability density [ns$^{-1}$]')
    
    t = np.linspace(-3 * sigma, tau2, 10000)
    y = ps1(t, p1, tau1, tau2)
    yc = ps1gauss(t, p1, tau1, tau2, sigma)
    yc2 = np.exp(logps1gauss(t, p1, tau1, tau2, sigma))
    y2 = ps2gauss(t, p1, tau1, tau2, min(tau1, tau2, sigma) / 100, sigma)
    yg = pgauss(t, sigma)
        
    mx = ps1gaussmax(p1, tau1, tau2, sigma)
    my = ps1gauss(mx, p1, tau1, tau2, sigma)
    
    ax.plot(t, y, label='S1')
    ax.plot(t, yc, label='S1 + res')
    ax.plot(t, yc2, '--', label='S1 + res (explog)')
    ax.plot(t, y2, ':', label='S2(T$\\approx$0) + res')
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

def check_ps2(p1=0.1, tau1=11, tau2=3200, T=15000, sigma=1000):
    """
    Plot ps2 and ps2gauss.
    """
    fig, ax = plt.subplots(num='ps12.check_ps2', clear=True)
    
    ax.set_title('Time distribution of S2 photons')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Probability density [arb. un.]')
    
    t = np.linspace(-3 * sigma, T + 3 * tau2, 10000)
    y = ps2(t, p1, tau1, tau2, T)
    yc = ps2gauss(t, p1, tau1, tau2, T, sigma)
    tg = np.linspace(-3 * sigma, 3 * sigma)
    yg = pgauss(tg, sigma)
    
    mx = ps2gaussmax(p1, tau1, tau2, T, sigma)
    my = ps2gauss(mx, p1, tau1, tau2, T, sigma)
    
    ax.plot(t, y / np.max(y), label='S2')
    ax.plot(tg, yg / np.max(yg), label='diffusion')
    ax.plot(t, yc / np.max(y), label='S2 + diffusion')
    ax.plot(mx, my / np.max(y), 'xk', label='maximum')
    
    ax.legend(loc='best')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')

    fig.tight_layout()
    fig.show()
    
    return fig

def check_gens1(p1=0.75, tau1=10, tau2=100, sigma=5):
    """
    Check gens12 with T = 0.
    """
    fig, ax = plt.subplots(num='pS1.check_gens1', clear=True)
    
    ax.set_title('Time distribution of S1 photons')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Probabilty density [ns$^{-1}$]')

    gen = np.random.default_rng(202012181733)
    s = gens12(100000, p1, tau1, tau2, 0, sigma, gen)
    # left, right = np.quantile(s, [0.001, 0.999])
    # s = s[(s >= left) & (s <= right)]
    
    ax.hist(s, bins='auto', histtype='step', density=True, label='samples histogram', zorder=2)
    
    t = np.linspace(np.min(s), np.max(s), 1000)
    y = ps1gauss(t, p1, tau1, tau2, sigma)
    
    ax.plot(t, y, label='pdf')
    
    p = 0.1
    l, r = ps12range(p1, tau1, tau2, 0, sigma, nsamples=len(s), p=p)
    ax.axvspan(l, r, color='#ddd', label=f'{p * 100:.2g} % bounds')
    
    ax.legend(loc='best')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')
    
    fig.tight_layout()
    fig.show()
    
    return fig

def check_gens2(p1=0.1, tau1=11, tau2=3200, T=15000, sigma=1000):
    """
    Check gens2 with T != 0.
    """
    fig, ax = plt.subplots(num='pS1.check_gens2', clear=True)
    
    ax.set_title('Time distribution of S2 photons')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Probabilty density [ns$^{-1}$]')

    gen = np.random.default_rng(202012181733)
    s = gens12(100000, p1, tau1, tau2, T, sigma, gen)
    
    ax.hist(s, bins='auto', histtype='step', density=True, label='samples histogram', zorder=2)
    
    t = np.linspace(np.min(s), np.max(s), 1000)
    y = ps2gauss(t, p1, tau1, tau2, T, sigma)
    
    ax.plot(t, y, label='pdf')
    
    p = 0.1
    l, r = ps12range(p1, tau1, tau2, T, sigma, nsamples=len(s), p=p)
    ax.axvspan(l, r, color='#ddd', label=f'{p * 100:.2g} % bounds')
    
    ax.legend(loc='best')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')
    
    fig.tight_layout()
    fig.show()
    
    return fig

if __name__ == '__main__':
    check_ps1()
    check_ps2()
    check_gens1()
    check_gens2()

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

DCR = 250 # (Hz) Dark count rate per PDM, 25 or 250
VL = 3 # fast/slow ratio, ER = 0.3, NR = 3
tauV = 7 # (ns) fast component tau
tauL = 1600 # (ns) slow component tau
T = 4 # (ms) time window
npdm = 8280 # number of PDMs
nphotons = 10 # (2-100) number of photons in the S1 signal
tres = 3 # (ns) temporal resolution (3-10)

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

def p_exp_gauss(t, tau, sigma):
    """
    Exponential pdf with scale tau convoluted with a normal with scale sigma.
    """
    return 1/tau * np.exp(-t/tau) * np.exp(1/2 * (sigma / tau) ** 2) * stats.norm.cdf(t / sigma - sigma / tau)

def p_S1_gauss(t, VL, tauV, tauL, sigma):
    """
    p_S1 convoluted with a normal with scale sigma.
    """
    V = 1 / (1 + 1 / VL)
    L = 1 / (1 + VL)
    return V * p_exp_gauss(t, tauV, sigma) + L * p_exp_gauss(t, tauL, sigma)

def check_p_S1(VL=3, tauV=7, tauL=1600, tres=10):
    """
    plot p_S1 and p_S1_gauss.
    """
    fig = plt.figure('temps1.check_p_S1')
    fig.clf()
    
    ax = fig.subplots(1, 1)
    
    ax.set_title('Time distribution of S1 photons')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Probabilty density [ns$^{-1}$]')
    
    t = np.linspace(-3 * tres, tauL, 10000)
    y = p_S1(t, VL, tauV, tauL)
    yc = p_S1_gauss(t, VL, tauV, tauL, tres)
    yg = p_gauss(t, tres)
    
    ax.plot(t, y, label='S1')
    ax.plot(t, yc, label='S1 + res')
    ax.plot(t, yg, label='res')
    
    ax.legend(loc='best')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')
    ax.set_yscale('log')
    ax.set_ylim(np.min(yc), np.max(y))

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
    fig = plt.figure('temps1.check_gen_S1')
    fig.clf()
    
    ax = fig.subplots(1, 1)
    
    gen = np.random.default_rng(202012181733)
    s = gen_S1(100000, VL, tauV, tauL, tres, gen)
    left, right = np.quantile(s, [0.001, 0.999])
    s = s[(s >= left) & (s <= right)]
    
    ax.hist(s, bins='auto', histtype='step', density=True, label='samples histogram')
    
    t = np.linspace(np.min(s), np.max(s), 10000)
    y = p_S1_gauss(t, VL, tauV, tauL, tres)
    
    ax.plot(t, y, label='pdf')
    
    ax.legend(loc='best')
    ax.grid()
    ax.set_yscale('log')
    
    fig.tight_layout
    fig.show()
    
    return fig

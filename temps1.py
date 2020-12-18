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
    return stats.expon.pdf(t, scale=tau)

def p_gauss(t, sigma):
    return stats.norm.pdf(t, scale=sigma)

def p_S1(t, VL, tauV, tauL):
    V = 1 / (1 + 1 / VL)
    L = 1 / (1 + VL)
    return V * p_exp(t, tauV) + L * p_exp(t, tauL)

def p_exp_gauss(t, tau, sigma):
    return 1/tau * np.exp(-t/tau) * np.exp(1/2 * (sigma / tau) ** 2) * stats.norm.cdf(t / sigma - sigma / tau)

def p_S1_gauss(t, VL, tauV, tauL, sigma):
    V = 1 / (1 + 1 / VL)
    L = 1 / (1 + VL)
    return V * p_exp_gauss(t, tauV, sigma) + L * p_exp_gauss(t, tauL, sigma)

def check_p_S1(VL=3, tauV=7, tauL=1600, tres=3):
    fig = plt.figure('temps1.check_p_S1')
    fig.clf()
    
    ax = fig.subplots(1, 1)
    
    t = np.linspace(-3 * tres, tauL, 10000)
    y = p_S1(t, VL, tauV, tauL)
    yc = p_S1_gauss(t, VL, tauV, tauL, tres)
    
    ax.plot(t, y, label='S1')
    ax.plot(t, yc, label='S1 + res')
    
    ax.legend(loc='best')
    ax.grid()
    ax.set_yscale('log')
    
    fig.tight_layout
    fig.show()
    
    return fig

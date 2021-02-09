import numpy as np
from matplotlib import pyplot as plt
import numba
from scipy import interpolate, signal

import pS1
import dcr
import filters
import textbox

def closenb(ref, nb, assume_sorted=False):
    """
    Return the array of elements from nb which are the closest to each element
    of ref.
    
    Parameters
    ----------
    ref : array (...)
        The reference values.
    nb : scalar or 1D array
        The neighbors.
    assume_sorted : bool
        If True the nb array is assumed to be sorted. Default False.
    
    Return
    ------
    out : int array (...)
        Array with the same shape of ref containing indices of elements of nb.
    """
    if not assume_sorted:
        nb = np.sort(nb)
    pos = np.searchsorted(nb, ref)
    posm1 = np.maximum(pos - 1, 0)
    pos = np.minimum(pos, len(nb) - 1)
    nbl = nb[posm1]
    nbh = nb[pos]
    assert np.all(ref - nbl) >= 0
    assert np.all(nbh - ref) >= 0
    return np.where(ref - nbl < nbh - ref, posm1, pos)

def testccfilter(nsignal=10, T=10000, rate=0.0025, VL=3, tauV=7, tauL=1600, tres=10, VLfilter=None, dt=1, offset=0, seed=None):
    """
    Plot filters output.
    """
    if VLfilter is None:
        VLfilter = VL
    
    if seed is None:
        seedgen = np.random.default_rng()
        seed = seedgen.integers(10001)
    generator = np.random.default_rng(seed)

    signal_loc = T / 10 + 5 * tres
    hits1 = pS1.gen_S1(nsignal, VL, tauV, tauL, tres, generator)
    hitdcr = dcr.gen_DCR((), T, rate, generator) - signal_loc
    
    hits = np.sort(np.concatenate([hits1, hitdcr]))
    
    mx = pS1.p_S1_gauss_maximum(VL, tauV, tauL, tres)
    ampl = pS1.p_S1_gauss(mx, VL, tauV, tauL, tres)
    fun = numba.njit('f8(f8)')(lambda t: pS1.p_S1_gauss(t + mx + offset, VL, tauV, tauL, tres) / ampl)
    left = -5 * tres
    right = 10 * tauL
    
    info = f"""\
nsignal = {nsignal}
T = {T}
rate = {rate}
VL = {VL}
tauV = {tauV}
tauL = {tauL}
tres = {tres}
VL filter = {VLfilter}
dt = {dt}
offset = {offset:.2g}
seed = {seed}"""
    
    interval = np.array([-tauL, T + 5 * tres]) - signal_loc
    t = np.arange(*interval, dt)

    v = filters.filter_cross_correlation(hits[None], t[None], fun, left, right)[0]
    interp = interpolate.interp1d(t, v)
    pidx, _ = signal.find_peaks(v, height=0.2)
    tpeak = t[pidx]
    hidx = closenb(tpeak, hits)
    thit = hits[hidx]
    
    fig, ax = plt.subplots(num='testccfilter1', clear=True)
    
    ax.plot(t, v, label='filter')
    ax.plot(hits1, interp(hits1), '.k', label='signal hits')
    ax.plot(hitdcr, interp(hitdcr), 'xk', label='noise hits')
    ax.plot(tpeak, interp(tpeak), 'o', color='#f55', label='peaks', zorder=-1)
    textbox.textbox(ax, info, loc='upper left', fontsize='small')
    
    ax.set_title('Cross correlation filter test')
    ax.set_xlabel('Time')
    ax.set_ylabel('Filter output')
    
    ax.legend(loc='upper right', fontsize='medium')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--')
    ax.grid(True, which='minor', linestyle=':')
    
    fig.tight_layout()
    fig.show()
    
    fig, axs = plt.subplots(2, 1, num='testccfilter2', clear=True, figsize=[6.4, 6.04])
    
    textbox.textbox(axs[0], info, loc='upper left', fontsize='small')
    axs[0].hist(thit - tpeak, bins='auto', histtype='step', label='time from peak to closest neighbor', zorder=2)
    axs[1].hist(interp(tpeak) - interp(thit), bins='auto', histtype='step', label='missing height to peak', zorder=2)
    
    axs[0].set_title('Cross correlation filter test')
    axs[0].set_xlabel('Time')
    axs[1].set_xlabel('Filter output')
    
    for ax in axs:
        ax.set_ylabel('Counts per bin')
        ax.legend(loc='upper right', fontsize='medium')
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='--')
        ax.grid(True, which='minor', linestyle=':')
    
    fig.tight_layout()
    fig.show()

if __name__ == '__main__':
    testccfilter()

import uproot
import numpy as np
from matplotlib import pyplot as plt

import qsigma

def plot_histogram(ax, counts, bins, **kw):
    """
    Plot an histogram.
    
    Parameters
    ----------
    ax : matplotlib axis
        The axis where the histogram is drawn.
    counts, bins : array
        The output from `np.histogram`.
    **kw :
        Keyword arguments are passed to `ax.plot`.
    
    Return
    ------
    lines : tuple
        The return value from `ax.plot`.
    """
    return ax.plot(np.concatenate([bins[:1], bins]), np.concatenate([[0], counts, [0]]), drawstyle='steps-post', **kw)

def hist2samples(counts, bins, gen=None):
    """
    Convert an histogram to an array of samples.
    
    The position of the samples in each bin is drawn at random uniformly.
    
    Parameters
    ----------
    counts, bins : array
        The output from `np.histogram`.
    gen : random generator, optional
        A numpy random number generator.
    
    Return
    ------
    x : array
        Samples which would yield the input histogram.
    """
    if gen is None:
        gen = np.random.default_rng()
    x = np.repeat(bins[:-1], counts)
    w = np.repeat(np.diff(bins), counts)
    x += w * gen.uniform(size=len(x))
    assert len(x) == np.sum(counts)
    return x

fig, axs = plt.subplots(2, 1, num='plot_saturation', clear=True, sharex=True, figsize=[6.4, 7.19])

axs[1].set_xlabel('Time [$\\mu$s]')
axs[0].set_ylabel('PE per 1/3 $\\mu$s per electron (or whatever)')
axs[1].set_ylabel('PE density per bin per electron [$\\mu$s$^{-1}$]')

root = uproot.open('plot_saturation.root')
for k in root.keys():
    th1d = root[k]
    counts, bins = th1d.numpy()
    int_counts = np.rint(counts / np.min(counts[counts != 0])).astype(int)
    x = hist2samples(int_counts, bins)
    sigma = qsigma.qsigma(x)
    label = k.decode() + f' $\\sigma_q$ = {sigma:.2g} $\\mu$s'
    plot_histogram(axs[0], counts, bins, label=label)
    norm_counts = counts / np.sum(counts / np.diff(bins))
    plot_histogram(axs[1], norm_counts, bins)

axs[0].legend()
axs[0].set_yscale('log')
for ax in axs:
    ax.minorticks_on()
    ax.grid(True, 'major', linestyle='--')
    ax.grid(True, 'minor', linestyle=':')

fig.tight_layout()
fig.show()

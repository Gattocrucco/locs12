from scipy import stats
import numpy as np

def sampling_bounds(dist, nsamples, p, p_right=None):
    """
    For a given distribution, compute the two endpoints such that the
    probability of having no sample below the first/above the second when
    drawing nsamples of them is p.
    
    Parameters
    ----------
    dist : str, tuple or scipy.stats.rv_continuous
        A distribution object. The only methods called are ppf and isf. If a
        string, it must be the name of a distribution in scipy.stats. If a
        tuple, the first item must be the name of a distribution, the others
        are passed as arguments.
    nsamples : int
        The number of samples.
    p : scalar
        The probability of getting samples below/above the computed endpoints.
    p_right : scalar, optional
        If given, p is the probability for the lower endpoint, p_right for the
        higher one.
    
    Return
    ------
    left, right : scalar
        The endpoints.
    """
    
    if isinstance(dist, str):
        dist = getattr(stats, dist)
    elif isinstance(dist, tuple):
        dist = getattr(stats, dist[0])(*dist[1:])
    p_left = p
    if p_right is None:
        p_right = p
    left  = dist.ppf(-np.expm1(np.log1p(-p_left ) / nsamples))
    right = dist.isf(-np.expm1(np.log1p(-p_right) / nsamples))
    return left, right

from scipy import stats
import numpy as np

def qsigma(sample, nsigma=1):
    # TODO compute the error on this quantity. Also, maybe investigate the
    # various measures of scale. There should be a lot in R.
    p = stats.norm.cdf([-nsigma, nsigma])
    q1, q2 = np.quantile(sample, p)
    return (q2 - q1) / 2

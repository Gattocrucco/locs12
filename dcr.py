import numpy as np

def gen_DCR(size, T, rate, generator=None):
    """
    Generate uniformly distributed hits. The output shape is size + number of
    events per time window T.
    """
    size = (size,) if np.isscalar(size) else tuple(size)
    if generator is None:
        generator = np.random.default_rng()
    size += (int(np.rint(T * rate)),)
    return generator.uniform(0, T, size)

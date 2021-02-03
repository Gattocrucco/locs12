import numpy as np

def named_cartesian_product(**fields):
    # TODO for completeness, add *args to do an unnamed cartesian product.
    # Only one of *args, **fields can be non-empty.
    
    fields = {k: np.asarray(v) for k, v in fields.items()}
    
    shape = sum((array.shape for array in fields.values()), start=())
    dtype = np.dtype([
        (name, array.dtype)
        for name, array in fields.items()
    ])
    out = np.empty(shape, dtype)
    
    offset = 0
    for name, array in fields.items():
        index = np.full(len(shape), None)
        length = len(array.shape)
        index[offset:offset + length] = slice(None)
        out[name] = array[tuple(index)]
        offset += length
    
    return out

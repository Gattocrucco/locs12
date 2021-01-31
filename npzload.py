import numpy as np

class NPZLoad:
    """
    Superclass for adding automatic serialization to/from npz files. Only
    scalar/array instance variables are considered.
    
    Instance methods
    ----------------
    save : save the object to a file.
    
    Class methods
    -------------
    load : read an instance form a file.
    """
    
    def save(self, filename):
        """
        Save the object to file as a `.npz` archive.
        """
        classdir = dir(type(self))
        variables = {
            n: x
            for n, x in vars(self).items()
            if n not in classdir
            and not n.startswith('__')
            and (np.isscalar(x) or isinstance(x, np.ndarray))
        }
        np.savez(filename, **variables)
    
    @classmethod
    def load(cls, filename):
        """
        Return an instance loading the object from a file which was written by
        `save`.
        """
        self = cls.__new__(cls)
        arch = np.load(filename)
        for n, x in arch.items():
            if x.shape == ():
                x = x.item()
            setattr(self, n, x)
        arch.close()
        return self

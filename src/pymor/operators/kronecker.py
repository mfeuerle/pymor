from pymor.operators.interface import Operator

from pymor.vectorarrays.kronecker import KronVectorSpace


class KronProductOperator(Operator):
    """
    Represents the Kronecker product 

        (A (x) B)

    of two liner |Operator| A and B. If A is an n x m matrix, and B a p x k matrix, 
        the Kronecker product (A (x) B) represents a np x mk matrix.

    Attributes
    ----------
    source
        |KronVectorSpace| of dimension p x n
    range
        |KronVectorSpace| of dimension k x m
    base_space
        Underling |VectorSpace|, used in `source` and `range`.

    Parameters
    ----------
    A
        |NumPy array| representing the first linear |Operator| as matrix
    B
        Some (arbitrary) linear |Operator|
    base_space
        The underling |VectorSpace| used in the |KronVectorSpace| for `source` and `range`. 
        If `None`, the default space, defined in |KronVectorSpace| will be used.
    """

    linear = True

    # A has to be a numpy matrix
    def __init__(self, A, B, base_space=None, source_id=None, range_id=None, name=None):
        assert B.linear
        self.source = KronVectorSpace(B.source.dim, A.shape[0], base_space, source_id)
        self.range  = KronVectorSpace(B.range.dim , A.shape[1], base_space, range_id)
        self.__auto_init(locals())

    def apply(self, U, mu=None):
        assert U in self.source
        BUA = self.range.empty(reserve=len(U))
        for i in range(0, len(U)):
            # does lincomb profit from sparse matrix?
            BUA._array[i] = self.B.apply(U._array[i], mu=mu).lincomb(self.A)
        return BUA
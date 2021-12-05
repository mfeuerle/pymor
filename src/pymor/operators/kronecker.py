from pymor.operators.interface import Operator

from pymor.vectorarrays.kronecker import KronVectorSpace


class KronProductOperator(Operator):
    r"""Class for the Kronecker product operator.

    This operator represents the Kronecker product (or Tensor product)

    .. math::
        A \otimes B(\mu) := 
        \begin{pmatrix}
        a_{1,1} B(\mu) & \dots & a_{1,m}B(\mu) \\
        \vdots &\ddots &\vdots \\
        a_{n,1} B(\mu) & \dots & a_{n,m} B(\mu) \end{pmatrix},
        \qquad
        A \otimes B(\mu) : \mathbb{R}^{np} \rightarrow \mathbb{R}^{mk},
    
    of a matrix operator :math:`A \in \mathbb{R}^{n \times m}` and a |Operator| :math:`B(\mu) : \mathbb{R}^{p} \rightarrow \mathbb{R}^{k}`.

    Attributes
    ----------
    source
        |KronVectorSpace| of dimension :math:`p \times n`
    range
        |KronVectorSpace| of dimension :math:`k \times m`
    base_space
        Underling |VectorSpace|, used in `source` and `range`.

    Parameters
    ----------
    A
        First matrix operator as |NumPy array|.
    B
        Second |Operator|.
    base_space
        The underling |VectorSpace| used in the |KronVectorSpace| for `source` and `range`. 
        If `None`, the default space, defined in |KronVectorSpace| will be used.
    """

    #linear = True

    # A has to be a numpy matrix
    def __init__(self, A, B, base_space=None, source_id=None, range_id=None, name=None):
        # assert B.linear   # code should work for nonlin too?
        self.source = KronVectorSpace(B.source.dim, A.shape[0], base_space, source_id)
        self.range  = KronVectorSpace(B.range.dim , A.shape[1], base_space, range_id)
        self.linear = B.linear
        self.__auto_init(locals())

    def apply(self, U, mu=None):
        assert U in self.source
        BUA = self.range.empty(reserve=len(U))
        for i in range(0, len(U)):
            # does lincomb profit from sparse matrix?
            BUA._array[i] = self.B.apply(U._array[i], mu=mu).lincomb(self.A)
        return BUA
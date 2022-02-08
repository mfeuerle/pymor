import numpy as np

from pymor.operators.interface import Operator
from pymor.vectorarrays.kronecker import KronVectorSpace
from pymor.algorithms.to_matrix import to_matrix


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
    
    of a matrix operator :math:`A \in \mathbb{R}^{n \times m}` and a linear |Operator| :math:`B(\mu) : \mathbb{R}^{p} \rightarrow \mathbb{R}^{k}`.

    .. todo::
        Support for sparse maticies A (limited due to VectorSpace.lincomb(coefs) expects numpy array).
        Support for parameter dependent operators A possible? (To avoid overhead, A.parametric==True could be usefull)
        Rule Tables for this Operator needed s.t. it can be transfered to a affine operator?

    Attributes
    ----------
    source
        |KronVectorSpace| of dimension :math:`p \times n`
    range
        |KronVectorSpace| of dimension :math:`k \times m`

    Parameters
    ----------
    A
        Linear non-parametric |Operator| suppertet by |to_matrix()| or |NumPy array| matrix representation.
    B
        Linear |Operator|.
    """

    linear = True

    def __init__(self, A, B, source_id=None, range_id=None, name=None):
        assert B.linear
        assert (isinstance(A, Operator) and A.parametric == False) or isinstance(A, np.ndarray)

        if isinstance(A, Operator):
            A = to_matrix(A)    # alternative: A = A.as_source_array().to_numpy()
            # if A.sparse: does lincomb profit from or even take sparse matricies?
        
        self.source = KronVectorSpace(B.source.dim, A.shape[1], B.source, source_id)
        self.range  = KronVectorSpace(B.range.dim , A.shape[0], B.range , range_id)
        self.__auto_init(locals())

    def apply(self, U, mu=None):
        assert U in self.source
        BUA = self.range.empty(reserve=len(U))
        for i in range(0, len(U)):
            # does lincomb profit from or even take sparse matricies?
            BUA.append(self.B.apply(U._array[i], mu=mu).lincomb(self.A))
        return BUA
    
    def as_source_array(self, mu=None):
        B = self.B.as_source_array(mu)
        AB = self.source.empty(reserve=self.range.dim)
        for j in range(0,self.A.shape[0]):
            for i in range(0,len(B)):
                AB.append(B[i].lincomb(np.reshape(self.A[j,:],(self.A.shape[1],1))))
        return AB
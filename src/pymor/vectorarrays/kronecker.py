import numpy as np
from numbers import Number

from pymor.vectorarrays.interface import VectorArray, VectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace

class KronVectorArray(VectorArray):
    r"""|VectorArray| implementation for |KronProductOperator|.

    |KronVectorArray| wraps an arbitray |VectorArray| for |KronProductOperator|.

    .. warning::
        This class is not intended to be instantiated directly. Use
        the associated |KronVectorSpace| instead.
    """
    def __init__(self, array, space):
        self._array = array
        self.space = space
        self._len = np.NaN  # make sure, constructor is never used alone

    def __len__(self):
        return self._len

    def __getitem__(self, ind):
        assert self.check_ind(ind)
        return self.space.make_array(self._array[ind], ensure_copy=False)

    def __delitem__(self, ind):
        assert self.check_ind_unique(ind)
        if type(ind) is slice:
            ind = set(range(*ind.indices(self._len)))
        elif not hasattr(ind, '__len__'):
            ind = {ind if 0 <= ind else self._len + ind}
        else:
            l = self._len
            ind = {i if 0 <= i else l+i for i in ind}
        self._array = np.delete(self._array,sorted(ind))
        self._len = len(self._array)
    
    def append(self, other, remove_from_other=False):
        if remove_from_other:
            self._array = np.append(self._array, other._array)
        else:
            self._array = np.append(self._array, other.copy()._array)
        self._len = len(self._array)

    def copy(self, deep=False):
        return self.space.make_array(self._array, ensure_copy=True, deep=deep)        

    def scal(self, alpha):
        if isinstance(alpha, Number):
            for i in range(0,self._len):
                self._array[i].scal(alpha)
        else:
            for i in range(0,self._len):
                self._array[i].scal(alpha[i])

    def axpy(self, alpha, x):
        assert isinstance(alpha, Number) \
            or isinstance(alpha, np.ndarray) and alpha.shape == (self._len,)
        assert x._len == 1 or x._len == self._len
        if isinstance(alpha, Number):
            if x._len == 1:
                for i in range(0,self._len):
                    self._array[i] = self._array[i].axpy(alpha,x._array[0])
            else:
                for i in range(0,self._len):
                    self._array[i].axpy(alpha,x._array[i])
        else:
            if x._len == 1:
                for i in range(0,self._len):
                    self._array[i] = self._array[i].axpy(alpha[i],x._array[0])
            else:
                for i in range(0,self._len):
                    self._array[i] = self._array[i].axpy(alpha[i],x._array[i])

    def lincomb(self, coefficients, _ind=None):
        assert 1<= len(coefficients.shape) <=2
        assert coefficients.shape[-1] == self._len
        if len(coefficients.shape) == 1:
            result = self.space.zeros(1)
            for j in range(0,len(coefficients)):
                result._array[0].axpy(coefficients[j], self._array[j])
        else:
            result = self.space.zeros(coefficients.shape[0])
            for i in range(0,coefficients.shape[0]):
                for j in range(0,coefficients.shape[1]):
                    result._array[i].axpy(coefficients[i,j], self._array[j])
        return result

    def _inner(self,x,y):
        return np.sum(x.pairwise_inner(y))

    def inner(self, other, product=None):
        if product is not None:
            return product.apply2(self, other)
        else:
            ip = np.zeros((self._len,other._len))
            for i in range(0,self._len):
                for j in range(0,other._len):
                    ip[i,j] = self._inner(self._array[i],other._array[j])
            return ip

    def pairwise_inner(self, other, product=None):
        assert self._len == other._len
        if product is not None:
            return product.pairwise_apply2(self, other)
        else:
            ip = np.zeros(self._len)
            for i in range(0,self._len):
                ip[i] = self._inner(self._array[i],other._array[i])
            return ip

    def _norm(self):
        return np.sqrt(self._norm2())

    def _norm2(self):
        return self.pairwise_inner(self)

    def dofs(self, dof_indices):
        raise NotImplementedError

    def amax(self):
        return NotImplementedError

    def to_numpy(self, ensure_copy=False):
        data = np.empty((self._len, self._array[0].dim, len(self._array[0])))
        for i in range(0,self._len):
            data[i] = self._array[i].to_numpy(ensure_copy=ensure_copy).T
        return data

    def __str__(self):
        return self._array[:self._len].__str__()

    def _format_repr(self, max_width, verbosity):
        return super()._format_repr(max_width, verbosity, override={'array': str(self._array[:self._len].__str__())})




class KronVectorSpace(VectorSpace):
    r"""|VectorSpace| wrapper for |KronProductOperator|.

    |KronVectorSpace| is designed to wrap an arbitrary |VectorSpace| for |KronProductOperator|.

    While |KronVectorSpace| represents vectors of dimension :math:`nm`, they are internally stored as 
    matricies of dimension :math:`n\times m` for efficient matrix-based operations within in 
    |KronProductOperator|. For storing those matrix-shaped vectors, some underling |VectorSpace| is used.

    Parameters
    ----------
    size1
        First dimension :math:`n` of the matrix-shaped :math:`n\times m` vectors.
    size2
        Second dimension :math:`m` of the matrix-shaped :math:`n\times m` vectors.
    base_space 
        |VectorSpace| used to store the matrix-shaped vectors.
        If `None`, |NumpyVectorSpace| is used.
    """
    def __init__(self, size1, size2, base_space=None, id=None):
        self.dim = size1*size2
        self.id = id
        self.size1 = size1
        self.size2 = size2
        if base_space is None:
            self.base_space = NumpyVectorSpace(size1)
        else:
            assert base_space.dim == size1
            self.base_space = base_space
        self.base_vector_type = type(self.base_space.zeros(count=self.size2))        

    def __eq__(self, other):
        # Just copied form |NumpyVectorSpace|.
        return type(other) is type(self) and self.size1 == other.size1 and self.size2 == other.size2 and self.id == other.id

    def __hash__(self):
        # Just copied form |NumpyVectorSpace|.
        return hash(self.dim) + hash(self.id)

    def empty(self, reserve=0):
        assert reserve >= 0
        va = KronVectorArray(np.empty(reserve, dtype=self.base_vector_type), self)
        va._len = reserve
        return va

    def zeros(self, count=1):
        assert count >= 0
        va = KronVectorArray(np.empty(count, dtype=self.base_vector_type), self)
        for i in range(0, count):
            va._array[i] =  self.base_space.zeros(count=self.size2)
        va._len = count
        return va

    def full(self, value, count=1):
        assert count >= 0
        va = KronVectorArray(np.empty(count, dtype=self.base_vector_type), self)
        for i in range(0, count):
            va._array[i] = self.base_space.full(value, count=self.size2)
        va._len = count
        return va

    def random(self, count=1, distribution='uniform', random_state=None, seed=None, **kwargs):
        assert count >= 0
        va = KronVectorArray(np.empty(count, dtype=self.base_vector_type), self)
        for i in range(0, count):
            va._array[i] = self.base_space.random(count=self.size2, distribution=distribution, random_state=random_state, seed=seed, reserve=0, **kwargs)
        va._len = count
        return va

    def from_numpy(self, data, ensure_copy=False):
        """
        Parameters
        ----------
        data
            |NumPy array| with `len(data.shape) == 2` for a single vector, 
            or `len(data.shape) == 3` for multiple vectors.
            For a single vector, we have `data.shape[0] = size1`, `data.shape[1] = size2`,
            for multiple vectors, `data.shape[0] = #vectors`, `data.shape[1] = size1`, `data.shape[2] = size2`.
        """
        if len(data.shape) == 2:
            assert data.shape[0] == self.size1 and data.shape[1] == self.size2
            va = KronVectorArray(np.empty(1, dtype=self.base_vector_type), self)
            va._array[0] = self.base_space.from_numpy(data.T.copy() if ensure_copy else data.T)
            va._len = 1
            return va
        elif len(data.shape) == 3:
            assert data.shape[1] == self.size1 and data.shape[2] == self.size2
            va = KronVectorArray(np.empty(data.shape[0], dtype=self.base_vector_type), self)
            for i in range(0,data.shape[0]):
                va._array[i] = self.base_space.from_numpy(data[i].T.copy() if ensure_copy else data[i].T)
            va._len = data.shape[0]
            return va
        else:
            raise IndexError

    def make_array(self, data, ensure_copy=False, deep=False):
        """
        Parameters
        ----------
        data
            Either a single |VectorArray| of the `base_space` or a |NumPy array| of |VectorArray|'s
        ensure_copy
            If `True`, `data` will be copied, otherwise, the ownership of `data` will be transferd to 
            the new |KronVectorArray| without making a copy.
            Default: `False`
        deep
            If `True`, the data will imideatly be copied, otherwise, the data will only be copied if necessary.
            Only significant, if `ensure_copy` is `True`.
            Default: `False`
        """
        if isinstance(data, self.base_vector_type):
            assert len(data) == self.size2
            va = KronVectorArray(np.empty(1, dtype=self.base_vector_type), self)
            va._array[0] = data.copy(deep=deep) if ensure_copy else data
            va._len = 1
        elif isinstance(data, np.ndarray) and isinstance(data[0], self.base_vector_type):
            va = KronVectorArray(np.empty(len(data), dtype=self.base_vector_type), self)
            for i in range(0,len(data)):
                assert len(data[i]) == self.size2
                va._array[i] = data[i].copy(deep=deep) if ensure_copy else data[i]
            va._len = len(data)
        else:
            raise IndexError
        if not ensure_copy:
            va.is_view = True
        return va
        

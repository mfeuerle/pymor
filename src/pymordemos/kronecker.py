import numpy as np
from pymor.basic import *
from pymor.operators.kronecker import KronProductOperator

n = 4
K = 2

A = np.random.randint(5, size=(K,K))
B = np.random.randint(5, size=(n,n))

A = 0.5*(A+A.T) + K*np.eye(K)
B = 0.5*(B+B.T) + n*np.eye(n)

B_ = NumpyMatrixOperator(B)
A_ = NumpyMatrixOperator(A)

# T = KronProductOperator(A,B_)
T = KronProductOperator(A_,B_)


print('\n')
print('Random test runs for Matrix-Vector product:')
for i in range(0, 5):
    X = np.random.rand(n,K)
    BXAt = np.matmul(np.matmul(B,X), A.T)
    
    x = T.source.from_numpy(X)
    Tx = T.apply(x)

    print(f'\tError: {np.linalg.norm(BXAt.T.reshape((1,-1))-Tx.to_numpy())}')

X = T.source.random(3)
X.lincomb(np.array([[1, 1, 1],[2, 2, 2]]))
X2 = T.source.random(5)
X.append(X2)
del X[2:6]

X = T.source.random(4)
x = np.reshape(X.to_numpy(), (len(X),n*K))

print('\n')
G = X.gramian()
g = np.dot(x,x.T)
print(f'Calculating inner product matrix, error vs np.dot: {np.linalg.norm(G-g)}')
print(G)

print('\n')
Q = gram_schmidt(X)
print('Orthogonalisation w.r.t. the euclidian product')
print(Q.gramian())

print('\n')
Q = gram_schmidt(X,product=T)
print('Orthogonalisation w.r.t. a TesorOperator product')
print(Q.gramian(product=T))
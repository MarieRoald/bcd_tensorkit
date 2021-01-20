from block_tenkit_admm import hierarchical_nnls
import numpy as np

def test_nnls():
    I, J = 10, 20
    rank = 5
    A = np.random.uniform(size=(I, rank))
    B = np.random.uniform(size=(J, rank))

    X = A@B.T

    B_hat = np.random.uniform(size=(J, rank))
    for i in range(5):
        B_hat = hierarchical_nnls.nnls(A, X, B_hat.T, 500).T

    assert np.linalg.norm(B_hat - B) < 1e-8


def test_prox_reg_nnls():
    I, J = 10, 20
    rank = 5
    A = np.random.uniform(size=(I, rank))
    B = np.random.uniform(size=(J, rank))

    X = A@B.T

    B_hat = np.random.uniform(size=(J, rank))
    for i in range(5):
        B_hat = hierarchical_nnls.prox_reg_nnls(A, X, B_hat.T, 0, 0*B_hat.T, 500).T

    assert np.linalg.norm(B_hat - B) < 1e-8

    B_center = np.random.uniform(size=(J, rank))
    B_hat = np.random.uniform(size=(J, rank))
    for i in range(50):
        B_hat = hierarchical_nnls.prox_reg_nnls(A, X, B_hat.T, 1, B_center.T, 100).T
    

    assert np.linalg.norm((A.T@A + np.eye(rank))@B_hat.T - (A.T@X  + B_center.T)) < 1e-5
    
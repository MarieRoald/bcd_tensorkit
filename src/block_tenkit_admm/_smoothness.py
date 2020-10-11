from textwrap import dedent
from warnings import warn

import numpy as np
import pyamg
from scipy import sparse
import scipy.sparse.linalg as spla
import scipy.linalg as sla


class _SmartSymmetricPDSolver:
    """Utility for when the same symmetric system will be solved many times.
    """
    supported_sparse = {"lu", "ilu", "amg", "amg_cg", "cg"}
    supported_dense = {"chol"}
    def __init__(self, system_of_eqs, method=None):
        """

        Arguments
        ---------

        """
        if method is None and sparse.issparse(system_of_eqs):
            method = "lu"
        elif method is None:
            method = "chol"
        
        method = method.lower()

        if (not sparse.issparse(system_of_eqs)) and (method not in self.supported_dense):
            warn(f"'{method}' is not supported for dense matrices. Resolving to 'chol'.")

        if method == "amg":
            system_of_eqs = sparse.csr_matrix(system_of_eqs)
            self.amg_grid = pyamg.smoothed_aggregation_solver(system_of_eqs)
        elif method == "amg_cg":
            system_of_eqs = sparse.csr_matrix(system_of_eqs)
            self.preconditioner = pyamg.smoothed_aggregation_solver(system_of_eqs).aspreconditioner()
        elif method == "lu":
            self.lu = spla.splu(system_of_eqs)
        elif method == "ilu_cg":
            self.spilu = spla.spilu(system_of_eqs)
            self.preconditioner = spla.LinearOperator(system_of_eqs.shape, lambda x: self.spilu.solve(x))
        elif method == "chol":
            self.chol = sla.cho_factor(system_of_eqs)
        elif method == "cg":
            self.preconditioner = None
        else:
            raise ValueError(dedent(
                f"""\
                Unsupported method {method}
                  * For dense matrices, choose one of {self.supported_dense}
                  * For sparse matrices, choose one of {self.supported_sparse}
                """
            ))
        
        self.system_of_eqs = system_of_eqs
        self.method = method
    
    def solve(self, x):
        if self.method == "chol":
            try:
                return sla.cho_solve(self.chol, x)
            except IndexError:
                raise IndexError(
                f"c: {self.chol[0].shape}\n"
                f"X: {x.shape}"
                )
        elif self.method == "lu":
            return self.lu.solve(x)
        elif self.method in {"cg", "amg_cg", "ilu_cg"}:
            if x.ndim > 1:
                return np.stack(
                    [self.solve(x[..., i]) for i in range(x.shape[-1])],
                    axis=-1
                )
            return spla.cg(self.system_of_eqs, x, M=self.preconditioner, tol=1e-10, maxiter=20)[0]
        elif self.method == "amg":
            if x.ndim > 1:
                return np.stack(
                    [self.solve(x[..., i]) for i in range(x.shape[-1])],
                    axis=-1
                )
                
            return self.amg_grid.solve(x, tol=1e-10)

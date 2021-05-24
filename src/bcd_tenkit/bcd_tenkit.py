# TODO: Prox class for each constraint
# TODO: Convergence checker class

from abc import ABC, abstractproperty, abstractmethod
import tenkit
from tenkit.decomposition import BaseDecomposer
from pathlib import Path
import numpy as np
from scipy import sparse
import scipy.linalg as sla
from tenkit import base
import h5py
from warnings import warn
from tenkit import utils
from condat_tv import tv_denoise_matrix

try:
    import cvxpy
except ImportError:
    pass

from tenkit.decomposition.cp import get_sse_lhs
from tenkit.decomposition.parafac2 import compute_projected_X as compute_projected_tensor
import tenkit.base
from ._smoothness import _SmartSymmetricPDSolver
from ._tv_prox import TotalVariationProx
from unimodal_regression import unimodal_regression
from .hierarchical_nnls import nnls, prox_reg_nnls

# TODO: Input random state for init
__all__ = [
    "BaseSubProblem", "Mode0RLS", "Mode0ADMM", "Mode2RLS", "Mode2ADMM",
    "DoubleSplittingParafac2ADMM", "SingleSplittingParafac2ADMM", 
    "FlexibleCouplingParafac2", "BCDCoupledMatrixDecomposer"
]

# Initialisation method doesn't affect the results much
# so we keep it uniform to have feasible init with NN constraints
INIT_METHOD_A = 'uniform'
INIT_METHOD_B = 'uniform'
INIT_METHOD_C = 'uniform'


class BaseSubProblem(ABC):
    def __init__(self):
        pass

    def init_subproblem(self, mode, auxiliary_variables):
        """Initialise the subproblem

        Arguments
        ---------
        mode : int
            The mode of the tensor that this subproblem should optimise wrt
        auxiliary_variable: list[dict]
            List of dictionaries, one dictionary per subproblem. All extra variables
            used by the subproblems should be stored in its corresponding auxiliary_variables
            dictionary
        rank : int
            The rank of the decomposition
        X : np.ndarray
            The dataset to fit against.
        """
        pass

    def update_decomposition(self, decomposition, auxiliary_variables):
        pass

    def regulariser(self, decomposition) -> float:
        return 0
    
    @property
    def checkpoint_params(self):
        return {}

    def load_from_hdf5_group(self, group):
        pass

    @abstractmethod
    def get_coupling_errors(self, decomposition):
        return []


def l2_ball_projection(factor_matrices):
    scale = np.maximum(1, np.linalg.norm(factor_matrices, axis=0, keepdims=True))
    return factor_matrices / scale


class Mode0RLS(BaseSubProblem):
    """
    Subproblem class used to update the first mode of an evolving tensor decomposition.
    
    Initialised before the decomposer, and init_subproblem is called before the first
    fitting iteration. The update_decomposition method is called each factor
    update iteration.
    """
    def __init__(self, non_negativity=False, ridge_penalty=0):
        self.non_negativity = non_negativity
        self.ridge_penalty = ridge_penalty

    def init_subproblem(self, mode, decomposer):
        """Initialise the subproblem

        Note that all extra variables used by this subproblem should be stored in the corresponding
        auxiliary_variables dictionary

        Arguments
        ---------
        mode : int
            The mode of the tensor that this subproblem should optimise wrt
        decomposer : block_tenkit_admm.double_splitting_parafac2.BCDCoupledMatrixDecomposer
            The decomposer that uses this subproblem
        """
        #assert mode == 0
        self.mode = 0
        self.X = decomposer.X
        self.unfolded_X = np.concatenate([X_slice for X_slice in self.X], axis=1)

    def _get_rightsolve(self):
        rightsolve = base.rightsolve
        if self.non_negativity:
            rightsolve = nnls
        if self.ridge_penalty:
            rightsolve = base.add_rightsolve_ridge(rightsolve, self.ridge_penalty)
        return rightsolve
        
    def update_decomposition(self, decomposition, auxiliary_variables):
        # TODO: This should all be fixed by _get_rightsolve
        if self.non_negativity:
            cross_product = 0
            factors_times_data = 0
            for k, Xk in enumerate(self.X):
                Bk = decomposition.B[k]
                Dk = np.diag(decomposition.C[k])
                factors_times_data += Xk @ Bk @ Dk
                cross_product += Dk @ Bk.T @ Bk @ Dk
            
            decomposition.A[...] = nnls(cross_product.T, factors_times_data.T, decomposition.A.T, 100).T
        else:
            rightsolve = self._get_rightsolve()
            right = np.concatenate([c*B for c, B in zip(decomposition.C, decomposition.B)], axis=0)
            decomposition.A[...] = rightsolve(right.T, self.unfolded_X)
    
    def regulariser(self, decomposition):
        if not self.ridge_penalty:
            return 0
        return self.ridge_penalty*np.sum(decomposition.A**2)
    
    def get_coupling_errors(self, decomposition):
        return []


class Mode0ADMM(BaseSubProblem):
    def __init__(self, ridge_penalty=0, l2_ball_constraint=False, non_negativity=False, max_its=10, tol=1e-5, rho=None, verbose=False):
        self.tol = tol
        self.non_negativity = non_negativity
        self.ridge_penalty = ridge_penalty
        self.max_its = max_its
        self.rho = rho 
        self.auto_rho = (rho is None)
        self.verbose = verbose
        self.l2_ball_constraint = l2_ball_constraint
    
    def init_subproblem(self, mode, decomposer):
        """Initialise the subproblem

        Note that all extra variables used by this subproblem should be stored in the corresponding
        auxiliary_variables dictionary

        Arguments
        ---------
        mode : int
            The mode of the tensor that this subproblem should optimise wrt
        decomposer : block_tenkit_admm.double_splitting_parafac2.BCDCoupledMatrixDecomposer
            The decomposer that uses this subproblem
        """
        init_method = getattr(np.random, INIT_METHOD_A)
        self.mode = 0
        self.X = decomposer.X
        self.unfolded_X = np.concatenate([X_slice for X_slice in self.X], axis=1)

        I = self.X[0].shape[0]
        rank = decomposer.rank
        self.aux_factor_matrix = init_method(size=(I, rank))
        self.dual_variables = init_method(size=(I, rank))
        decomposer.auxiliary_variables[0]['aux_factor_matrix'] = self.aux_factor_matrix
        decomposer.auxiliary_variables[0]['dual_variables'] = self.dual_variables
        #self._update_aux_factor_matrix(decomposer.decomposition)

    def update_decomposition(self, decomposition, auxiliary_variables):
        self.cache = {}
        self._recompute_normal_equation(decomposition)
        self._set_rho(decomposition)
        self._recompute_cholesky_cache(decomposition)

        for i in range(self.max_its):
            self._update_factor(decomposition)
            self._update_aux_factor_matrix(decomposition)
            self._update_duals(decomposition)

            if self._has_converged(decomposition):
                break
        
        self.num_its = i
    
    def regulariser(self, decomposition):
        regulariser = 0
        if self.ridge_penalty:
            regulariser += self.ridge_penalty*np.sum(decomposition.A**2)
        return regulariser

    def get_coupling_errors(self, decomposition):
        A = decomposition.A
        return [np.linalg.norm(A - self.aux_factor_matrix)/np.linalg.norm(A)]
    
    def _recompute_normal_equation(self, decomposition):
        self.cache["normal_eq_lhs"] = 0
        self.cache["normal_eq_rhs"] = 0
        for k, Bk in enumerate(decomposition.B):
            BkDk = Bk*decomposition.C[k, np.newaxis, :]
            self.cache["normal_eq_lhs"] += BkDk.T@BkDk
            self.cache["normal_eq_rhs"] += self.X[k]@BkDk

    def _set_rho(self, decomposition):
        if not self.auto_rho:
            return
        else:
            normal_eq_lhs = self.cache['normal_eq_lhs']
            self.rho = np.trace(normal_eq_lhs)/decomposition.rank

    def _recompute_cholesky_cache(self, decomposition):
        # Prepare to compute choleskys
        lhs = self.cache['normal_eq_lhs']
        I = np.eye(decomposition.rank)
        self.cache['cholesky'] = sla.cho_factor(lhs + (0.5*self.rho + self.ridge_penalty)*I)

    def _update_factor(self, decomposition):
        rhs = self.cache['normal_eq_rhs']  # X{kk}
        chol_lhs = self.cache['cholesky']  # L{kk}
        rho = self.rho  # rho(kk)

        prox_rhs = rhs + rho/2*(self.aux_factor_matrix - self.dual_variables)
        decomposition.factor_matrices[self.mode][...]= sla.cho_solve(chol_lhs, prox_rhs.T).T

    def _update_duals(self, decomposition):
        self.previous_dual_variables = self.dual_variables
        self.dual_variables = self.dual_variables + decomposition.factor_matrices[self.mode] - self.aux_factor_matrix
    
    def _update_aux_factor_matrix(self, decomposition):
        self.previous_factor_matrix = self.aux_factor_matrix

        perturbed_factor = decomposition.factor_matrices[self.mode] + self.dual_variables
        if self.l2_ball_constraint:
            self.aux_factor_matrix[:] = l2_ball_projection(perturbed_factor)
        elif self.non_negativity:
            self.aux_factor_matrix[:] = np.maximum(perturbed_factor, 0)
        elif self.ridge_penalty:
            # min (r/2)||X||^2 + (rho/2)|| X - Y ||^2
            # min (r/2) Tr(X^TX) + (rho/2)Tr(X^TX) - rho Tr(X^TY) + (rho/2) Tr(Y^TY)
            # Differentiate wrt X:
            # rX + rho (X - Y) = 0
            # rX + rho X = rho Y
            # (r + rho) X = rho Y
            # X = (rho / (r + rho)) Y

            # The ridge penalty is now imposed on the data fitting term instead
            self.aux_factor_matrix[:] = perturbed_factor
            #self.aux_factor_matrix = (self.rho / (self.ridge_penalty + self.rho)) * perturbed_factor
        else:
            self.aux_factor_matrix[:] = perturbed_factor
        pass
    
    def _has_converged(self, decomposition):
        factor_matrix = decomposition.factor_matrices[self.mode]
        coupling_error = np.linalg.norm(factor_matrix-self.aux_factor_matrix)**2
        coupling_error /= np.linalg.norm(self.aux_factor_matrix)**2
        
        aux_change = np.linalg.norm(self.aux_factor_matrix-self.previous_factor_matrix)**2
        
        dual_var_norm = np.linalg.norm(self.dual_variables)**2
        aux_change_criterion = (aux_change + 1e-16) / (dual_var_norm + 1e-16)
        if self.verbose:
            print("primal criteria", coupling_error, "dual criteria", aux_change)
        return coupling_error < self.tol and aux_change_criterion < self.tol


class Mode0ProjectedADMM(BaseSubProblem):
    def __init__(self, projection_problem_idx, ridge_penalty=0, l1_penalty=0, non_negativity=False, max_its=10, tol=1e-5, rho=None, verbose=False):
        self.tol = tol
        self.non_negativity = non_negativity
        self.ridge_penalty = ridge_penalty
        self.max_its = max_its
        self.rho = rho 
        self.auto_rho = (rho is None)
        self.verbose = verbose
        self.l1_penalty = l1_penalty
        self.projection_problem_idx = projection_problem_idx
    
    def init_subproblem(self, mode, decomposer):
        """Initialise the subproblem

        Note that all extra variables used by this subproblem should be stored in the corresponding
        auxiliary_variables dictionary

        Arguments
        ---------
        mode : int
            The mode of the tensor that this subproblem should optimise wrt
        decomposer : block_tenkit_admm.double_splitting_parafac2.BCDCoupledMatrixDecomposer
            The decomposer that uses this subproblem
        """
        init_method = getattr(np.random, INIT_METHOD_A)
        self.mode = 0
        self.unfolded_X = np.concatenate([X_slice for X_slice in decomposer.X], axis=1)

        I = decomposer.X[0].shape[0]
        rank = decomposer.rank
        self.aux_factor_matrix = init_method(size=(I, rank))
        self.dual_variables = init_method(size=(I, rank))
        decomposer.auxiliary_variables[0]['aux_factor_matrix'] = self.aux_factor_matrix
        decomposer.auxiliary_variables[0]['dual_variables'] = self.dual_variables
        
        self.B_aux_vars = decomposer.auxiliary_variables[self.projection_problem_idx]

    def update_decomposition(self, decomposition, auxiliary_variables):
        self.cache = {}
        self._recompute_normal_equation(decomposition)
        self._set_rho(decomposition)
        self._recompute_cholesky_cache(decomposition)

        for i in range(self.max_its):
            self._update_factor(decomposition)
            self._update_aux_factor_matrix(decomposition)
            self._update_duals(decomposition)

            if self._has_converged(decomposition):
                break
        
        self.num_its = i
    
    def regulariser(self, decomposition):
        regulariser = 0
        if self.ridge_penalty:
            regulariser += self.ridge_penalty*np.sum(decomposition.A**2)
        return regulariser

    def get_coupling_errors(self, decomposition):
        A = decomposition.A
        return [np.linalg.norm(A - self.aux_factor_matrix)/np.linalg.norm(A)]
    
    def _recompute_normal_equation(self, decomposition):
        self.cache["normal_eq_lhs"] = 0
        self.cache["normal_eq_rhs"] = 0
        C = decomposition.C
        B = self.B_aux_vars['blueprint_B']
        projected_tensor = self.B_aux_vars['projected_tensor']
        for k, Ck in enumerate(C):
            BDk = B*C[k, np.newaxis, :]
            self.cache["normal_eq_lhs"] += BDk.T@BDk
            self.cache["normal_eq_rhs"] += projected_tensor[..., k]@BDk

    def _set_rho(self, decomposition):
        if not self.auto_rho:
            return
        else:
            normal_eq_lhs = self.cache['normal_eq_lhs']
            self.rho = np.trace(normal_eq_lhs)/decomposition.rank

    def _recompute_cholesky_cache(self, decomposition):
        # Prepare to compute choleskys
        lhs = self.cache['normal_eq_lhs']
        I = np.eye(decomposition.rank)
        self.cache['cholesky'] = sla.cho_factor(lhs + (0.5*self.rho + self.ridge_penalty)*I)

    def _update_factor(self, decomposition):
        rhs = self.cache['normal_eq_rhs']  # X{kk}
        chol_lhs = self.cache['cholesky']  # L{kk}
        rho = self.rho  # rho(kk)

        prox_rhs = rhs + rho/2*(self.aux_factor_matrix - self.dual_variables)
        decomposition.factor_matrices[self.mode][...]= sla.cho_solve(chol_lhs, prox_rhs.T).T

    def _update_duals(self, decomposition):
        self.previous_dual_variables = self.dual_variables
        self.dual_variables = self.dual_variables + decomposition.factor_matrices[self.mode] - self.aux_factor_matrix
    
    def _update_aux_factor_matrix(self, decomposition):
        self.previous_factor_matrix = self.aux_factor_matrix

        perturbed_factor = decomposition.factor_matrices[self.mode] + self.dual_variables
        if self.non_negativity:
            self.aux_factor_matrix[:] = np.maximum(perturbed_factor, 0)
        elif self.ridge_penalty:
            # The ridge penalty is now imposed on the data fitting term instead
            self.aux_factor_matrix[:] = perturbed_factor
        else:
            self.aux_factor_matrix[:] = perturbed_factor
    
    def _has_converged(self, decomposition):
        factor_matrix = decomposition.factor_matrices[self.mode]
        coupling_error = np.linalg.norm(factor_matrix-self.aux_factor_matrix)**2
        coupling_error /= np.linalg.norm(self.aux_factor_matrix)**2
        
        aux_change = np.linalg.norm(self.aux_factor_matrix-self.previous_factor_matrix)**2
        
        dual_var_norm = np.linalg.norm(self.dual_variables)**2
        aux_change_criterion = (aux_change + 1e-16) / (dual_var_norm + 1e-16)
        if self.verbose:
            print("primal criteria", coupling_error, "dual criteria", aux_change)
        return coupling_error < self.tol and aux_change_criterion < self.tol


class Mode2RLS(BaseSubProblem):
    def __init__(self, non_negativity=False, ridge_penalty=0):
        self.non_negativity = non_negativity
        self.ridge_penalty = ridge_penalty

    
    def init_subproblem(self, mode, decomposer):
        """Initialise the subproblem

        Note that all extra variables used by this subproblem should be stored in the corresponding
        auxiliary_variables dictionary

        Arguments
        ---------
        mode : int
            The mode of the tensor that this subproblem should optimise wrt
        decomposer : block_tenkit_admm.double_splitting_parafac2.BCDCoupledMatrixDecomposer
            The decomposer that uses this subproblem
        """
        #assert mode == 2
        self.mode = 2
        self.X = decomposer.X

    def _get_rightsolve(self):
        rightsolve = base.rightsolve
        if self.non_negativity:
            rightsolve = base.non_negative_rightsolve
        if self.ridge_penalty:
            rightsolve = base.add_rightsolve_ridge(rightsolve, self.ridge_penalty)
        return rightsolve
        
    def update_decomposition(self, decomposition, auxiliary_variables):
        # TODO: This should all be fixed by _get_rightsolve
        if self.non_negativity:
            for k, (c_row, factor_matrix) in enumerate(zip(decomposition.C,  decomposition.B)):
                X_k_vec = self.X[k].reshape(-1, 1)
                lhs = base.khatri_rao(decomposition.A, factor_matrix)
                c_row[...] = nnls(lhs, X_k_vec, c_row, 100)
            return

        rightsolve = self._get_rightsolve()
        for k, (c_row, factor_matrix) in enumerate(zip(decomposition.C,  decomposition.B)):
            X_k_vec = self.X[k].reshape(-1, 1)
            lhs = base.khatri_rao(decomposition.A, factor_matrix)
            c_row[...] = rightsolve(lhs.T, X_k_vec.T.ravel())

    def regulariser(self, decomposition):
        if not self.ridge_penalty:
            return 0
        return self.ridge_penalty*np.sum(decomposition.C**2)

    def get_coupling_errors(self, decomposition):
        return []


class Mode2ADMM(BaseSubProblem):
    def __init__(self, ridge_penalty=0, l2_ball_constraint=False, non_negativity=False, max_its=10, tol=1e-5, rho=None, verbose=False):
        self.tol = tol
        self.ridge_penalty = ridge_penalty
        self.non_negativity = non_negativity
        self.max_its = max_its
        self.rho = rho 
        self.auto_rho = (rho is None)
        self.verbose = verbose
        self.l2_ball_constraint = l2_ball_constraint
    
    def init_subproblem(self, mode, decomposer):
        """Initialise the subproblem

        Note that all extra variables used by this subproblem should be stored in the corresponding
        auxiliary_variables dictionary

        Arguments
        ---------
        mode : int
            The mode of the tensor that this subproblem should optimise wrt
        decomposer : block_tenkit_admm.double_splitting_parafac2.BCDCoupledMatrixDecomposer
            The decomposer that uses this subproblem
        """
        init_method = getattr(np.random, INIT_METHOD_C)
        self.mode = 2
        self.X = decomposer.X
        self.unfolded_X = np.concatenate([X_slice for X_slice in self.X], axis=1)

        K = len(decomposer.X)
        rank = decomposer.rank
        self.aux_factor_matrix = init_method(size=(K, rank))
        self.dual_variables = init_method(size=(K, rank))
        decomposer.auxiliary_variables[self.mode]['aux_factor_matrix'] = self.aux_factor_matrix
        decomposer.auxiliary_variables[self.mode]['dual_variables'] = self.dual_variables
        #self._update_aux_factor_matrix(decomposer.decomposition)

    def update_decomposition(self, decomposition, auxiliary_variables):
        self.cache = {}
        self._recompute_normal_equation(decomposition)
        self._set_rho(decomposition)
        self._recompute_cholesky_cache(decomposition)

        for i in range(self.max_its):
            self._update_factor(decomposition)
            self._update_aux_factor_matrix(decomposition)
            self._update_duals(decomposition)

            if self._has_converged(decomposition):
                break
        
        self.num_its = i
    
    def regulariser(self, decomposition):
        regulariser = 0
        if self.ridge_penalty:
            regulariser += self.ridge_penalty*np.sum(decomposition.C**2)
        return regulariser

    def get_coupling_errors(self, decomposition):
        C = decomposition.C
        return [np.linalg.norm(C - self.aux_factor_matrix)/np.linalg.norm(C)]

    def _recompute_normal_equation(self, decomposition):
        A = decomposition.A
        B = decomposition.B
        AtA = decomposition.A.T@decomposition.A
        K = len(self.X)
        self.cache["normal_eq_lhs"] = [AtA*(Bk.T@Bk) for Bk in B]
        self.cache["normal_eq_rhs"] = [np.diag(A.T@self.X[k]@B[k]) for k in range(K)]

    def _set_rho(self, decomposition):
        if not self.auto_rho:
            return
        else:
            normal_eq_lhs = self.cache['normal_eq_lhs']
            self.rho = [np.trace(lhs)/decomposition.rank for lhs in normal_eq_lhs]

    def _recompute_cholesky_cache(self, decomposition):
        # Prepare to compute choleskys
        I = np.eye(decomposition.rank)
        self.cache['cholesky'] = [sla.cho_factor(lhs + (0.5*rho + self.ridge_penalty)*I) for rho,lhs in zip(self.rho, self.cache['normal_eq_lhs'])]

    def _update_factor(self, decomposition):
        rhs = self.cache['normal_eq_rhs']  # X{kk}
        chol_lhs = self.cache['cholesky']  # L{kk}
        rho = self.rho  # rho(kk)

        K = len(rhs)
        for k in range(K):
            prox_rhs = rhs[k] + rho[k]/2*(self.aux_factor_matrix[k] - self.dual_variables[k])
            decomposition.factor_matrices[self.mode][k] = sla.cho_solve(chol_lhs[k], prox_rhs.T).T

    def _update_duals(self, decomposition):
        self.previous_dual_variables = self.dual_variables
        self.dual_variables = self.dual_variables + decomposition.factor_matrices[self.mode] - self.aux_factor_matrix
    
    def _update_aux_factor_matrix(self, decomposition):
        self.previous_factor_matrix = self.aux_factor_matrix

        perturbed_factor = decomposition.factor_matrices[self.mode] + self.dual_variables
        if self.non_negativity and self.l2_ball_constraint:
            for r in range(perturbed_factor.shape[1]):
                x = cvxpy.Variable(perturbed_factor.shape[0])
                objective = cvxpy.Minimize(cvxpy.sum_squares(x - perturbed_factor[:, r]))
                constraints = [0 <= x, cvxpy.sum_squares(x) <= 1]
                problem = cvxpy.Problem(objective, constraints)
                problem.solve()
                self.aux_factor_matrix[:, r] = x.value
        elif self.non_negativity:
            self.aux_factor_matrix[:] =  np.maximum(perturbed_factor, 0)
        elif self.l2_ball_constraint:
            self.aux_factor_matrix[:] = l2_ball_projection(perturbed_factor)
        # Ridge is not incorporated in the aux factor matrix
        #elif self.ridge_penalty:
        #    # min (r/2)||X||^2 + (rho/2)|| X - Y ||^2
        #    # min (r/2) Tr(X^TX) + (rho/2)Tr(X^TX) - rho Tr(X^TY) + (rho/2) Tr(Y^TY)
        #    # Differentiate wrt X:
        #    # rX + rho (X - Y) = 0
        #    # rX + rho X = rho Y
        #    # (r + rho) X = rho Y
        #    # X = (rho / (r + rho)) Y
        #    for k, _ in enumerate(perturbed_factor):
        #        self.aux_factor_matrix[k] = (self.rho[k] / (self.ridge_penalty + self.rho[k])) * perturbed_factor[k]
        else:
            self.aux_factor_matrix[:] = perturbed_factor
    
    def _has_converged(self, decomposition):
        factor_matrix = decomposition.factor_matrices[self.mode]
        coupling_error = np.linalg.norm(factor_matrix-self.aux_factor_matrix)
        coupling_error /= np.linalg.norm(self.aux_factor_matrix)
        
        aux_change = np.linalg.norm(self.aux_factor_matrix-self.previous_factor_matrix)
        
        dual_var_norm = np.linalg.norm(self.dual_variables)
        aux_change_criterion = (aux_change + 1e-16) / (dual_var_norm + 1e-16)
        if self.verbose:
            print("primal criteria", coupling_error, "dual criteria", aux_change)
        return coupling_error < self.tol and aux_change_criterion < self.tol


class Mode2ProjectedADMM(BaseSubProblem):
    def __init__(self, projection_problem_idx, ridge_penalty=0, l1_penalty=0, non_negativity=False, max_its=10, tol=1e-5, rho=None, verbose=False):
        self.tol = tol
        self.non_negativity = non_negativity
        self.ridge_penalty = ridge_penalty
        self.max_its = max_its
        self.rho = rho 
        self.auto_rho = (rho is None)
        self.verbose = verbose
        self.l1_penalty = l1_penalty
        self.projection_problem_idx = projection_problem_idx
    
    def init_subproblem(self, mode, decomposer):
        """Initialise the subproblem

        Note that all extra variables used by this subproblem should be stored in the corresponding
        auxiliary_variables dictionary

        Arguments
        ---------
        mode : int
            The mode of the tensor that this subproblem should optimise wrt
        decomposer : block_tenkit_admm.double_splitting_parafac2.BCDCoupledMatrixDecomposer
            The decomposer that uses this subproblem
        """
        init_method = getattr(np.random, INIT_METHOD_A)
        self.mode = 2

        K = len(decomposer.X)
        rank = decomposer.rank
        self.aux_factor_matrix = init_method(size=(K, rank))
        self.dual_variables = init_method(size=(K, rank))
        decomposer.auxiliary_variables[2]['aux_factor_matrix'] = self.aux_factor_matrix
        decomposer.auxiliary_variables[2]['dual_variables'] = self.dual_variables
        
        self.B_aux_vars = decomposer.auxiliary_variables[self.projection_problem_idx]

    def update_decomposition(self, decomposition, auxiliary_variables):
        self.cache = {}
        self._recompute_normal_equation(decomposition)
        self._set_rho(decomposition)
        self._recompute_cholesky_cache(decomposition)

        for i in range(self.max_its):
            self._update_factor(decomposition)
            self._update_aux_factor_matrix(decomposition)
            self._update_duals(decomposition)

            if self._has_converged(decomposition):
                break
        
        self.num_its = i
    
    def regulariser(self, decomposition):
        regulariser = 0
        if self.ridge_penalty:
            regulariser += self.ridge_penalty*np.sum(decomposition.C**2)
        return regulariser

    def get_coupling_errors(self, decomposition):
        C = decomposition.C
        return [np.linalg.norm(C - self.aux_factor_matrix)/np.linalg.norm(C)]
    
    def _recompute_normal_equation(self, decomposition):
        A = decomposition.A
        B = self.B_aux_vars['blueprint_B']
        projected_tensor = self.B_aux_vars['projected_tensor']
        unfolded_X = projected_tensor.reshape(-1, projected_tensor.shape[-1]).T
        self.cache["normal_eq_lhs"] = (A.T@A) * (B.T@B)
        self.cache["normal_eq_rhs"] = unfolded_X @ tenkit.base.khatri_rao(A, B)

    def _set_rho(self, decomposition):
        if not self.auto_rho:
            return
        else:
            normal_eq_lhs = self.cache['normal_eq_lhs']
            self.rho = np.trace(normal_eq_lhs)/decomposition.rank

    def _recompute_cholesky_cache(self, decomposition):
        # Prepare to compute choleskys
        lhs = self.cache['normal_eq_lhs']
        I = np.eye(decomposition.rank)
        self.cache['cholesky'] = sla.cho_factor(lhs + (0.5*self.rho + self.ridge_penalty)*I)

    def _update_factor(self, decomposition):
        rhs = self.cache['normal_eq_rhs']  # X{kk}
        chol_lhs = self.cache['cholesky']  # L{kk}
        rho = self.rho  # rho(kk)

        prox_rhs = rhs + rho/2*(self.aux_factor_matrix - self.dual_variables)
        decomposition.factor_matrices[self.mode][...]= sla.cho_solve(chol_lhs, prox_rhs.T).T

    def _update_duals(self, decomposition):
        self.previous_dual_variables = self.dual_variables
        self.dual_variables = self.dual_variables + decomposition.factor_matrices[self.mode] - self.aux_factor_matrix
    
    def _update_aux_factor_matrix(self, decomposition):
        self.previous_factor_matrix = self.aux_factor_matrix

        perturbed_factor = decomposition.factor_matrices[self.mode] + self.dual_variables
        if self.non_negativity and self.l1_penalty:
            return np.maximum(self.aux_factor_matrix - 2*self.l1_penalty/self.rho, 0)
        elif self.l1_penalty:
            return np.sign(self.aux_factor_matrix)*np.maximum(np.abs(self.aux_factor_matrix) - 2*self.l1_penalty/self.rho, 0)
        elif self.non_negativity:
            self.aux_factor_matrix =  np.maximum(perturbed_factor, 0)
        elif self.ridge_penalty:
            # The ridge penalty is now imposed on the data fitting term instead
            self.aux_factor_matrix = perturbed_factor
        else:
            self.aux_factor_matrix = perturbed_factor
        pass
    
    def _has_converged(self, decomposition):
        factor_matrix = decomposition.factor_matrices[self.mode]
        coupling_error = np.linalg.norm(factor_matrix-self.aux_factor_matrix)**2
        coupling_error /= np.linalg.norm(self.aux_factor_matrix)**2
        
        aux_change = np.linalg.norm(self.aux_factor_matrix-self.previous_factor_matrix)**2
        
        dual_var_norm = np.linalg.norm(self.dual_variables)**2
        aux_change_criterion = (aux_change + 1e-16) / (dual_var_norm + 1e-16)
        if self.verbose:
            print("primal criteria", coupling_error, "dual criteria", aux_change)
        return coupling_error < self.tol and aux_change_criterion < self.tol


def _random_orthogonal_matrix(num_rows, num_cols):
    return np.linalg.qr(np.random.standard_normal(size=(num_rows, num_cols)))[0]


class DoubleSplittingParafac2ADMM(BaseSubProblem):
    def __init__(
        self,
        rho=None,
        auto_rho_scaling=1,
        tol=1e-3,
        max_its=5,

        non_negativity=False,
        l2_similarity=None,
        ridge_penalty=None,
        l1_penalty=None,
        tv_penalty=None,
        verbose=False,
        l2_solve_method=None,
        use_preinit=False,
        scad_penalty=None,
        scad_parameter=3.7,
        unimodality=False,

        pf2_prox_iterations=1,
        
        compute_projected_tensor=False
    ):
        if rho is None:
            self.auto_rho = True
        else:
            self.auto_rho = False
        self.auto_rho_scaling = auto_rho_scaling

        self.rho = rho
        self.tol = tol
        self.max_its = max_its
     
        self.non_negativity = non_negativity
        self.l2_similarity = l2_similarity
        self.l1_penalty = l1_penalty
        self.tv_penalty = tv_penalty
        self.ridge_penalty = ridge_penalty
        self.unimodality = unimodality
        if self.ridge_penalty is None:
            self.ridge_penalty = 0

        self.l2_solve_method = l2_solve_method

        self.use_preinit = use_preinit
        self.scad_penalty = scad_penalty
        self.scad_parameter = scad_parameter
        self.pf2_prox_iterations = pf2_prox_iterations

        self._cache = {}
        self.compute_projected_tensor = compute_projected_tensor  # Only used for other modes

    def init_subproblem(self, mode, decomposer):
        """Initialise the subproblem

        Note that all extra variables used by this subproblem should be stored in the corresponding
        auxiliary_variables dictionary

        Arguments
        ---------
        mode : int
            The mode of the tensor that this subproblem should optimise wrt
        decomposer : block_tenkit_admm.double_splitting_parafac2.BCDCoupledMatrixDecomposer
            The decomposer that uses this subproblem
        """
        init_method = getattr(np.random, INIT_METHOD_B)

        K = len(decomposer.X)
        rank = decomposer.rank
        X = decomposer.X

        self.mode = 1
        self.X = decomposer.X

        self._cache['rho'] = [np.inf]*K

        self.blueprint_B = init_method(size=(rank, rank))
        self.projection_matrices = [np.eye(X[k].shape[1], rank) for k in range(K)]
        self.reg_Bks = [init_method(size=(X[k].shape[1], rank)) for k in range(K)]
        self.dual_variables_reg = [init_method(size=(X[k].shape[1], rank)) for k in range(K)]
        self.dual_variables_pf2 = [init_method(size=(X[k].shape[1], rank)) for k in range(K)]


        auxiliary_variables = decomposer.auxiliary_variables
        if 'blueprint_B' in auxiliary_variables[1] and self.use_preinit:
            self.blueprint_B[:] = auxiliary_variables[1]['blueprint_B']
        else:
            auxiliary_variables[1]['blueprint_B'] = self.blueprint_B

        if 'projection_matrices' in auxiliary_variables[1] and self.use_preinit:
            self.projection_matrices = auxiliary_variables[1]['projection_matrices']
        else:
            auxiliary_variables[1]['projection_matrices'] = self.projection_matrices
        auxiliary_variables[1]['reg_Bks'] = self.reg_Bks
        auxiliary_variables[1]['dual_variables_reg'] = self.dual_variables_reg
        auxiliary_variables[1]['dual_variables_pf2'] = self.dual_variables_pf2

        if self.compute_projected_tensor:
            self.projected_tensor = compute_projected_tensor(self.projection_matrices, X)
            auxiliary_variables[1]['projected_tensor'] = self.projected_tensor

    def update_smoothness_proxes(self):
        if self.l2_similarity is None:
            return

        if sparse.issparse(self.l2_similarity):
            I = sparse.eye(self.l2_similarity.shape[0])
        else:
            I = np.identity(self.l2_similarity.shape[0])
        
        reg_matrices = [self.l2_similarity + 0.5*rho*I for rho in self._cache['rho']]

        self._cache['l2_reg_solvers'] = [
            _SmartSymmetricPDSolver(reg_matrix, method=self.l2_solve_method)
            for reg_matrix in reg_matrices
        ]

    def update_decomposition(
        self, decomposition, auxiliary_variables
    ):
        self._cache = {}
        self.recompute_normal_equation(decomposition)
        self.set_rhos(decomposition)
        self.recompute_cholesky_cache(decomposition)
        self.update_smoothness_proxes()
        # breakpoint()
        # The decomposition is modified inplace each iteration
        for i in range(self.max_its):
            # Update Bks and Pks
            # Update blueprint
            # Update reg matrices
            # Update duals
            # print("unconstrained:", np.linalg.norm(decomposition.B[0], axis=0))
            self.update_unconstrained(decomposition)
            # print("unconstrained:", np.linalg.norm(decomposition.B[0], axis=0))

            for _ in range(self.pf2_prox_iterations):
                self.update_projections(decomposition)

                # print("blueprint:", np.linalg.norm(self.blueprint_B, axis=0))
                self.update_blueprint(decomposition)
                # print("blueprint:", np.linalg.norm(self.blueprint_B, axis=0))

            # print("reg", np.linalg.norm(self.reg_Bks[0], axis=0))
            self.update_reg_factors(decomposition)
            # print("reg", np.linalg.norm(self.reg_Bks[0], axis=0))

            # print("dual reg", np.linalg.norm(self.dual_variables_reg[0], axis=0))
            # print("dual pf2", np.linalg.norm(self.dual_variables_pf2[0], axis=0))
            self.update_duals(decomposition)
            # print("dual reg", np.linalg.norm(self.dual_variables_reg[0], axis=0))
            # print("dual pf2", np.linalg.norm(self.dual_variables_pf2[0], axis=0))

            if self.has_converged(decomposition):
                break
        
        if self.compute_projected_tensor:
            compute_projected_tensor(self.projection_matrices, self.X, out=self.projected_tensor)
        self.num_its = i

    def recompute_normal_equation(self, decomposition):
        K = decomposition.C.shape[0]
        self._cache['normal_eq_rhs'] = [None for _ in range(K)]
        self._cache['normal_eq_lhs'] = [None for _ in range(K)]
        for k in range(K):
            ADk = decomposition.A*decomposition.C[k, np.newaxis]
            self._cache['normal_eq_rhs'][k] = self.X[k].T@ADk
            self._cache['normal_eq_lhs'][k] = ADk.T@ADk

    def set_rhos(self, decomposition):
        if not self.auto_rho:
            self._cache['rho'] = [self.rho for _ in decomposition.B]
        else:
            normal_eq_lhs = self._cache['normal_eq_lhs']
            self._cache['rho'] = [self.auto_rho_scaling*np.trace(lhs)/decomposition.rank for lhs in normal_eq_lhs]

    def recompute_cholesky_cache(self, decomposition):
        # Prepare to compute choleskys
        rhos = self._cache['rho']
        normal_eq_lhs = self._cache['normal_eq_lhs']
        I = np.eye(decomposition.rank)
        self._cache['choleskys'] = [sla.cho_factor(lhs + (rho + self.ridge_penalty)*I) for rho, lhs in zip(rhos, normal_eq_lhs)]

    def update_unconstrained(self, decomposition):
        K = decomposition.C.shape[0]
        blueprint_B = self.blueprint_B  # DeltaB

        for k in range(K):
            rhs = self._cache['normal_eq_rhs'][k]  # X{kk}
            chol_lhs = self._cache['choleskys'][k]  # L{kk}
            rho = self._cache['rho'][k]  # rho(kk)
            
            P = self.projection_matrices[k]  # P{kk}
            reg_Bk = self.reg_Bks[k]  # ZB{kk}
            dual_variables_reg = self.dual_variables_reg[k]  # mu_B_Z{kk}
            dual_variables_pf2 = self.dual_variables_pf2[k]  # mu_DeltaB[kk]

            prox_rhs = rhs + 0.5*rho*(reg_Bk - dual_variables_reg) + 0.5*rho*(P@blueprint_B - dual_variables_pf2)
            decomposition.B[k][...] = sla.cho_solve(chol_lhs, prox_rhs.T).T

    def update_projections(self, decomposition):
        self.previous_projections = [pk for pk in self.projection_matrices]
        K = decomposition.C.shape[0]
        blueprint_B = self.blueprint_B  # DeltaB
        for k in range(K):
            Bk = decomposition.B[k]
            dual_variable_pf2_k = self.dual_variables_pf2[k]
            
            U, s, Vh = sla.svd((Bk + dual_variable_pf2_k)@blueprint_B.T, full_matrices=False)
            self.projection_matrices[k] = U@Vh

    def update_blueprint(self, decomposition):
        self.previous_blueprint_B = self.blueprint_B
        K = decomposition.C.shape[0]
        rho = self._cache['rho']
        involved_variables = zip(rho, self.projection_matrices, decomposition.B, self.dual_variables_pf2)
        self.blueprint_B[:] = sum(rho_k * Pk.T@(Bk + dual_pf2_k) for rho_k, Pk, Bk, dual_pf2_k in involved_variables)/np.sum(rho)
    
    def update_reg_factors(self, decomposition):
        self.previous_reg_Bks = [bk for bk in self.reg_Bks]
        Bks = decomposition.B
        dual_vars = self.dual_variables_reg
        self.reg_Bks = [self.prox(Bk + dual_var_reg, k) for k, (Bk, dual_var_reg) in enumerate(zip(Bks, dual_vars))]
    
    def prox(self, factor_matrix, k):
        # TODO: Comments
        # TODO: Check compatibility between different proxes
        # The above todo is not necessary if we implement a separate prox-class.
        rho = self._cache['rho'][k]
        if self.non_negativity and self.l1_penalty:
            return np.maximum(factor_matrix - 2*self.l1_penalty/rho, 0)
        elif self.l1_penalty:
            return np.sign(factor_matrix)*np.maximum(np.abs(factor_matrix) - 2*self.l1_penalty/rho, 0)
        elif self.unimodality and self.non_negativity:
            return unimodal_regression(factor_matrix, non_negativity=True)
        elif self.unimodality:
            return unimodal_regression(factor_matrix, non_negativity=False)
        elif self.tv_penalty:
            return tv_denoise_matrix(factor_matrix.T, self.tv_penalty/rho).T
        elif self.non_negativity:
            return np.maximum(factor_matrix, 0)      
        elif self.l2_similarity is not None:
            # Solve (2L + rho I)x = y -> (L + 0.5*rho I)x = 0.5*rho*y
            reg_solver = self._cache['l2_reg_solvers'][k]
            rhs = 0.5*rho*factor_matrix
            return reg_solver.solve(rhs)
        elif self.scad_penalty is not None:
            a = self.scad_parameter

            abs_fm = np.abs(factor_matrix)
            soft_threshold_mask = abs_fm < 2*self.scad_penalty
            intermediate_mask = (abs_fm < a*self.scad_penalty) & (~ soft_threshold_mask)

            # Unregularised indices are unchanged
            out = factor_matrix.copy()

            # Intermediate region
            inter_sel = factor_matrix[intermediate_mask]
            out[intermediate_mask] = (
                ((a - 1)*inter_sel - np.sign(inter_sel)*a*self.scad_penalty)
                /(a - 2)
            )

            # Soft thresholding region:
            st_sel = factor_matrix[soft_threshold_mask]
            out[soft_threshold_mask] = (
                np.sign(st_sel)*np.maximum(np.abs(st_sel) - self.scad_penalty, 0)
            )
            return factor_matrix
        else:
            return factor_matrix
    
    def update_duals(self, decomposition):
        K = decomposition.C.shape[0]
        for k in range(K):
            self.dual_variables_reg[k] += decomposition.B[k] - self.reg_Bks[k]
            self.dual_variables_pf2[k] += decomposition.B[k] - self.projection_matrices[k]@self.blueprint_B

    def has_converged(self, decomposition):       
        K = decomposition.C.shape[0]
        
        relative_reg_coupling_error = 0
        relative_pf2_coupling_error = 0

        relative_reg_change = 0
        relative_pf2_change = 0

        for k in range(K):
            relative_reg_coupling_error += (
                np.linalg.norm(decomposition.B[k] - self.reg_Bks[k])
                /(np.linalg.norm(decomposition.B[k]) * K)
            )
            relative_pf2_coupling_error += (
                np.linalg.norm(decomposition.B[k] - self.projection_matrices[k]@self.blueprint_B)
                /(np.linalg.norm(decomposition.B[k]) * K)
            )
            #sum_factor += np.linalg.norm(decomposition.B[k])

            relative_reg_change += (
                np.linalg.norm(self.previous_reg_Bks[k] - self.reg_Bks[k])
                /(np.linalg.norm(self.dual_variables_reg[k]) * K)
            )
            #sum_dual_reg += np.linalg.norm(self.dual_variables_reg)

            previous_pf2_Bks = self.previous_projections[k]@self.previous_blueprint_B
            pf2_Bk = self.projection_matrices[k]@self.blueprint_B
            relative_pf2_change += (
                np.linalg.norm(previous_pf2_Bks - pf2_Bk)
                /(np.linalg.norm(self.dual_variables_pf2[k]) * K)
            )
            #sum_dual_pf2 += np.linalg.norm(self.dual_variables_pf2)

        #relative_reg_change /= (sum_dual_reg + 1e-16)/K
        #relative_pf2_change /= (sum_dual_pf2 + 1e-16)/K
        relative_change_criterion = max(relative_pf2_change, relative_reg_change)

        #relative_reg_coupling_error = reg_coupling_error/sum_factor/K
        #relative_pf2_coupling_error = pf2_coupling_error/sum_factor/K
        relative_coupling_criterion = max(relative_pf2_coupling_error, relative_reg_coupling_error)


        return relative_change_criterion < self.tol and relative_coupling_criterion < self.tol

    def get_coupling_errors(self, decomposition):
        K = len(self.projection_matrices)
        pf2_coupling_error = 0
        reg_coupling_error = 0
        for k in range(K):
            pf2_Bk = self.projection_matrices[k]@self.blueprint_B
            
            pf2_coupling_error += np.linalg.norm(pf2_Bk - decomposition.B[k])/np.linalg.norm(decomposition.B[k])
            reg_coupling_error += np.linalg.norm(self.reg_Bks[k] - decomposition.B[k])/np.linalg.norm(decomposition.B[k])

        return reg_coupling_error, pf2_coupling_error

    def regulariser(self, decomposition):
        regulariser = 0
        if self.l2_similarity is not None:
            B = decomposition.B
            W = self.l2_similarity
            K = decomposition.C.shape[0]
            regulariser += sum(
                np.trace(B[k].T@W@B[k]) 
                for k in range(K) 
            )

        if self.tv_penalty is not None:
            for factor in decomposition.B:
                regulariser += TotalVariationProx(factor, self.tv_penalty).center_penalty()

        if self.ridge_penalty is not None:
            B = decomposition.B
            regulariser += self.ridge_penalty*sum(np.linalg.norm(Bk)**2 for Bk in B)

        if self.l1_penalty is not None:
            B = decomposition.B
            regulariser += self.l1_penalty*sum(np.sum(np.abs(Bk)) for Bk in B)

        if self.scad_penalty is not None:
            a = self.scad_parameter
            for Bk in decomposition.B:
                abs_fm = np.abs(Bk)

                # Masks for the three regions of the scad penalty
                soft_threshold_mask = abs_fm < 2*self.scad_penalty
                intermediate_mask = (abs_fm < a*self.scad_penalty) & (~ soft_threshold_mask)
                unregularised = (~intermediate_mask) & (~soft_threshold_mask)

                # The unregularised (constant penalty) part
                regulariser += unregularised.sum()*(a+1)*(self.scad_penalty**2)*0.5
                
                # The intermediate region
                inter_sel = abs_fm[intermediate_mask]
                regulariser += np.sum(
                    2*a*self.scad_penalty*inter_sel 
                    - inter_sel**2 
                    - self.scad_penalty**2
                )/(2*(a-1))

                # The soft-thresholding (LASSO) region
                st_sel = abs_fm[soft_threshold_mask]
                regulariser += self.scad_penalty * st_sel.sum()
            

        return regulariser


class SingleSplittingParafac2ADMM(BaseSubProblem):
    def __init__(
        self,
        rho=None,
        auto_rho_scaling=1,
        tol=1e-3,
        max_its=5,

        non_negativity=False,
        l2_similarity=None,
        ridge_penalty=None,
        l1_penalty=None,
        tv_penalty=None,
        verbose=False,
        l2_solve_method=None,
        use_preinit=True,
    ):
        if rho is None:
            self.auto_rho = True
        else:
            self.auto_rho = False
        self.auto_rho_scaling = auto_rho_scaling

        self.rho = rho
        self.tol = tol
        self.max_its = max_its
     
        self.non_negativity = non_negativity
        self.l2_similarity = l2_similarity
        self.l1_penalty = l1_penalty
        self.ridge_penalty = ridge_penalty
        self.tv_penalty = tv_penalty
        if self.ridge_penalty is None:
            self.ridge_penalty = 0

        self.l2_solve_method = l2_solve_method

        self.use_preinit = use_preinit

        self._cache = {}

    def init_subproblem(self, mode, decomposer):
        """Initialise the subproblem

        Note that all extra variables used by this subproblem should be stored in the corresponding
        auxiliary_variables dictionary

        Arguments
        ---------
        mode : int
            The mode of the tensor that this subproblem should optimise wrt
        decomposer : block_tenkit_admm.double_splitting_parafac2.BCDCoupledMatrixDecomposer
            The decomposer that uses this subproblem
        """
        init_method = getattr(np.random, INIT_METHOD_B)
        K = len(decomposer.X)
        rank = decomposer.rank
        X = decomposer.X

        self.mode = mode
        self.X = decomposer.X

        self._cache['rho'] = [np.inf]*K

        self.blueprint_B = init_method(size=(rank, rank))
        self.projection_matrices = [np.eye(X[k].shape[1], rank) for k in range(K)]
        self.reg_Bks = [init_method(size=(X[k].shape[1], rank)) for k in range(K)]
        self.dual_variables = [init_method(size=(X[k].shape[1], rank)) for k in range(K)]

        auxiliary_variables = decomposer.auxiliary_variables
        if 'blueprint_B' in auxiliary_variables[1] and self.use_preinit:
            self.blueprint_B[:] = auxiliary_variables[1]['blueprint_B']
        else:
            auxiliary_variables[1]['blueprint_B'] = self.blueprint_B

        if 'projection_matrices' in auxiliary_variables[1] and self.use_preinit:
            self.projection_matrices = auxiliary_variables[1]['projection_matrices']
        else:
            auxiliary_variables[1]['projection_matrices'] = self.projection_matrices
        auxiliary_variables[1]['reg_Bks'] = self.reg_Bks
        auxiliary_variables[1]['dual_variables'] = self.dual_variables

        self.update_projected_tensor(decomposer.decomposition, auxiliary_variables)

    def update_decomposition(
        self, decomposition, auxiliary_variables
    ):
        previous_reg_Bks = self._cache.get(
            'previous_reg_Bks', [reg_Bk.copy() for reg_Bk in self.reg_Bks]
        )
        self._cache = {}
        self._cache['previous_reg_Bks'] = previous_reg_Bks
        self.recompute_normal_equation(decomposition)
        self.set_rho(decomposition)
        self.recompute_cholesky_cache(decomposition)
        self.update_smoothness_prox()
        # breakpoint()
        # The decomposition is modified inplace each iteration            
        for i in range(self.max_its):
            self.update_projections(decomposition)
            self.update_projected_tensor(decomposition, auxiliary_variables)
            self.update_blueprint(decomposition)
            
            # This is ran iteration since it is needed for reg factors and duals
            self.update_decomposition_Bs(decomposition)  

            self.update_reg_factors(decomposition)
            self.update_duals(decomposition)

            if self.has_converged(decomposition):
                for k, reg_Bk in enumerate(self.reg_Bks):
                    self._cache['previous_reg_Bks'][k][:] = reg_Bk
                break
            
            for k, reg_Bk in enumerate(self.reg_Bks):
                self._cache['previous_reg_Bks'][k][:] = reg_Bk
            

        self.num_its = i
    
    def recompute_normal_equation(self, decomposition):
        A = decomposition.A
        C = decomposition.C
        self._cache['normal_eq_lhs'] = (A.T@A) * (C.T@C)

    def set_rho(self, decomposition):
        self._cache['rho'] = np.trace(self._cache['normal_eq_lhs'])

    def recompute_cholesky_cache(self, decomposition):
        # Prepare to compute choleskys
        K = decomposition.C.shape[0]
        rho = self._cache['rho']
        lhs = self._cache['normal_eq_lhs']
        I = np.eye(decomposition.rank)
        self._cache['cholesky'] = sla.cho_factor(lhs + (0.5*K*rho + self.ridge_penalty)*I)

    def update_projections(self, decomposition):
        """
        
        """
        A = decomposition.A
        blueprint_B = self.blueprint_B
        C = decomposition.C
        rho = self._cache['rho']
        for k, X_k in enumerate(self.X):
            unreg_lhs = (A*C[k])@(blueprint_B.T)
            reg_lhs = np.sqrt(rho/2)*(blueprint_B.T)
            lhs = np.vstack((unreg_lhs, reg_lhs))

            unreg_rhs = X_k
            reg_rhs = np.sqrt(rho/2)*(self.reg_Bks[k] - self.dual_variables[k]).T
            rhs = np.vstack((unreg_rhs, reg_rhs))
            
            self.projection_matrices[k][:] = base.orthogonal_solve(lhs, rhs).T
    
    def update_projected_tensor(self, decomposition, auxiliary_matrices):
        auxiliary_matrices[self.mode]['projected_tensor'] = tenkit.decomposition.parafac2.compute_projected_X(
            self.projection_matrices,
            self.X,
            out=auxiliary_matrices[self.mode].get('projected_tensor', None)
        )
        self._cache['projected_tensor'] = auxiliary_matrices[self.mode]['projected_tensor']

        self._cache['normal_eq_rhs'] = base.matrix_khatri_rao_product(
            self._cache['projected_tensor'],
            [decomposition.A, self.blueprint_B, decomposition.C],
            mode=1
        )

    def update_blueprint(self, decomposition):
        r"""
        ' -> transpose
        .khat. -> Khatri rao
        Y -> projected X unfolded along second mode

        || (A .khat. C) B' - Y || + \sum_k 0.5 \rho ||(P_k B - (Breg_k - U_k))'||^2

        M = (A .khat .C)

        Differentiate,
        Traces are implicit (which is ok since traces are invariant to cyclic permutations)

        2M' M B' - 2M' Y + \sum_k \rho (P_k' P_k B) - P_k' (Breg_k - U_k)) = 0
        (M'M B' + K \rho I) B' = M' Y + \sum_k \rho P_k' (Breg_k - U_k)
        """
        chol_lhs = self._cache['cholesky']  # L L' = (A'A * C'C) + 0.5 * rho * K *I
        rho = self._cache['rho']
        rhs = self._cache['normal_eq_rhs'].copy()     # Y(1) (A .khat. C)
        reg_Bks = self.reg_Bks
        dual_variables = self.dual_variables
        for projection_matrix, reg_Bk, dual_matrix in zip(self.projection_matrices, reg_Bks, dual_variables):
            rhs += 0.5*rho*projection_matrix.T@(reg_Bk - dual_matrix)

        self.blueprint_B[:] = sla.cho_solve(chol_lhs, rhs.T).T   

    def update_decomposition_Bs(self, decomposition):
        """Insert Pk B for B in the evolving tensor decomposition.
        """
        for Bk, projection_matrix in zip(decomposition.B, self.projection_matrices):
            Bk[:] = projection_matrix @ self.blueprint_B

    def update_reg_factors(self, decomposition):
        self.previous_reg_Bks = [bk for bk in self.reg_Bks]
        Bks = decomposition.B  # This is updated in the update_blueprint method
        dual_vars = self.dual_variables
        merged_Bks = np.concatenate(Bks, axis=1)
        merged_dual_vars = np.concatenate(dual_vars, axis=1)
        proxed_values = self.prox(merged_Bks + merged_dual_vars)

        for k, reg_Bk in enumerate(self.reg_Bks):
            reg_Bk[:] = proxed_values[:, k*decomposition.rank:(k+1)*decomposition.rank]

    def update_duals(self, decomposition):
        K = decomposition.C.shape[0]
        for k in range(K):
            self.dual_variables[k] += decomposition.B[k] - self.reg_Bks[k]

    def prox(self, factor_matrix):
        # TODO: Comments
        # TODO: Check compatibility between different proxes
        # The above todo is not necessary if we implement a separate prox-class.
        rho = self._cache['rho']
        if self.non_negativity and self.l1_penalty:
            return np.maximum(factor_matrix - 2*self.l1_penalty/rho, 0)
        elif self.tv_penalty:
            return tv_denoise_matrix(factor_matrix.T, self.tv_penalty/rho).T
            #return total_variation_prox(factor_matrix, 2*self.tv_penalty/rho)
        elif self.non_negativity:
            return np.maximum(factor_matrix, 0)
        elif self.l1_penalty:
            return np.sign(factor_matrix)*np.maximum(np.abs(factor_matrix) - 2*self.l1_penalty/rho, 0)
        elif self.l2_similarity is not None:
            # Solve (2L + rho I)x = y -> (L + 0.5*rho I)x = 0.5*rho*y
            reg_solver = self._cache['l2_reg_solver']
            rhs = 0.5*rho*factor_matrix
            return reg_solver.solve(rhs)
        else:
            return factor_matrix

    def update_smoothness_prox(self):
        if self.l2_similarity is None:
            return

        if sparse.issparse(self.l2_similarity):
            I = sparse.eye(self.l2_similarity.shape[0])
        else:
            I = np.identity(self.l2_similarity.shape[0])
        
        rho = self._cache['rho']
        reg_matrix = self.l2_similarity + 0.5*rho*I

        self._cache['l2_reg_solver'] = _SmartSymmetricPDSolver(reg_matrix, method=self.l2_solve_method)

    def has_converged(self, decomposition):   
        K = decomposition.C.shape[0]
        
        relative_coupling_criterion = 0
        relative_change_criterion = 0

        for k in range(K):
            relative_coupling_criterion += (
                np.linalg.norm(decomposition.B[k] - self.reg_Bks[k])
                /(np.linalg.norm(decomposition.B[k]) * K)
            )

            previous_reg_Bks = self._cache['previous_reg_Bks']
            relative_change_criterion += (
                np.linalg.norm(previous_reg_Bks[k] - self.reg_Bks[k])
                /(np.linalg.norm(self.dual_variables[k]) * K)
            )

        return relative_change_criterion < self.tol and relative_coupling_criterion < self.tol

    def get_coupling_errors(self, decomposition):
        K = len(self.projection_matrices)
        coupling_error = 0
        for k in range(K):
            factor_diff = self.reg_Bks[k] - decomposition.B[k]
            coupling_error += np.linalg.norm(factor_diff)/np.linalg.norm(decomposition.B[k])

        return [coupling_error]

    def regulariser(self, decomposition):
        regulariser = 0
        if self.l2_similarity is not None:
            B = decomposition.B
            W = self.l2_similarity
            K = decomposition.C.shape[0]
            regulariser += sum(
                np.trace(B[k].T@W@B[k]) 
                for k in range(K) 
            )

        if self.tv_penalty is not None:
            for factor in decomposition.B:
                regulariser += TotalVariationProx(factor, self.tv_penalty).center_penalty()

        if self.ridge_penalty is not None:
            B = decomposition.B
            regulariser += sum(np.linalg.norm(Bk)**2 for Bk in B)

        return regulariser


class FlexibleCouplingParafac2(BaseSubProblem):
    def __init__(
        self,
        mu_level=0.2,
        max_mu=1e12,
        mu_increase=1.05,
        num_projection_its=5,
        max_nnls_its=100,
        non_negativity=True,
        l2_penalty=None
    ):
        assert mu_increase > 1
        assert max_mu > 0
        assert mu_level > 0

        self.mu_level = mu_level
        self.max_mu = max_mu
        self.mu_increase = mu_increase
        self.num_projection_its = num_projection_its
        self.max_nnls_its = max_nnls_its
        self.non_negativity = non_negativity
        self.l2_penalty = l2_penalty
    
    def init_subproblem(self, mode, decomposer):
        """Initialise the subproblem

        Note that all extra variables used by this subproblem should be stored in the corresponding
        auxiliary_variables dictionary

        Arguments
        ---------
        mode : int
            The mode of the tensor that this subproblem should optimise wrt
        decomposer : block_tenkit_admm.double_splitting_parafac2.BCDCoupledMatrixDecomposer
            The decomposer that uses this subproblem
        """
        #assert mode == 1

        rank = decomposer.rank
        X = decomposer.X
        self.K = len(decomposer.X)
        self.mode = 1
        self.X = decomposer.X
        self.X_slice_norms_sq = [np.linalg.norm(Xk)**2 for Xk in self.X]
        self.X_norm_sq = sum(self.X_slice_norms_sq)

        self.current_iteration = 0

        # Initialise coupling strength:
        # mu[k] = 0.1 * ||X[k] - AD[k]B[k]^T||^2 / ||B[k]||^2
        dec = decomposer.decomposition
        slice_wise_sse = [np.linalg.norm(self.X[k] - dec.construct_slice(k))**2 for k in range(self.K)]
        self.mu = np.array(
            [slice_wise_sse[k]/(10*np.linalg.norm(dec.B[k])**2) for k in range(self.K)]
        )

        self.blueprint_B = np.random.uniform(size=(rank, rank))
        self.projection_matrices = [np.eye(X[k].shape[1], rank) for k in range(self.K)]

        decomposer.auxiliary_variables[1]['blueprint_B'] = self.blueprint_B
        decomposer.auxiliary_variables[1]['projection_matrices'] = self.projection_matrices

    def update_decomposition(
        self, decomposition, auxiliary_variables
    ):
        # Increase coupling strength
        self.mu *= self.mu_increase
        self.mu[self.mu > self.max_mu] = self.max_mu

        # Update projections and blueprint
        self.update_blueprint_and_projections(decomposition)
        
        # Update the factor matrices
        self.update_B(decomposition)

        # Reinitialise coupling strength after first iteration
        if self.current_iteration == 0:
            slice_wise_sse = [np.linalg.norm(self.X[k] - decomposition.construct_slice(k))**2 for k in range(self.K)]
            coupling_error = [np.linalg.norm(decomposition.B[k] - self.projection_matrices[k]@self.blueprint_B)**2 for k in range(self.K)]
            self.mu[:] = [slice_wise_sse[k] / (self.mu[k]*coupling_error[k]) for k in range(self.K)]

        self.current_iteration += 1

    def update_blueprint_and_projections(self, decomposition):
        for i in range(self.num_projection_its):
            B_mean = 0
            for k in range(self.K):
                B_mean += self.mu[k] * (self.projection_matrices[k].T@decomposition.B[k])
            self.blueprint_B[:] = B_mean / np.sum(self.mu)
            self.blueprint_B[:] /= np.linalg.norm(self.blueprint_B, keepdims=True)

            for k in range(self.K):
                U, s, Vh = sla.svd(decomposition.B[k] @ self.blueprint_B.T, full_matrices=False)
                self.projection_matrices[k][:] = U@Vh
    
    def update_B(self, decomposition):
        for k in range(self.K):
            ADk = decomposition.A * decomposition.C[k, np.newaxis]
            PkB = self.projection_matrices[k]@self.blueprint_B

            if self.non_negativity:
                decomposition.B[k][:] = prox_reg_nnls(ADk, self.X[k], decomposition.B[k].T, self.mu[k], PkB.T, self.max_nnls_its).T
            elif self.l2_penalty is not None:
                #raise NotImplementedError
                # || (ADk) Bk^T - Xk ||^2 + µ ||Bk^T - (PkB)^T||^2 + <Bk, W Bk>
                # Notation: <X, Y> = Tr(X^TY)
                # <(ADk) Bk^T, (ADk) Bk^T> - 2 <(ADk) Bk^T, Xk> + µ<Bk^T, Bk^T> - 2 µ <Bk^T, (PkB)^T> + <Bk, W Bk>
                # Differentiate wrt Bk^T and set equal to zero
                # 2 (ADk)^T (ADk) Bk^T - 2 (ADk)^T Xk + 2µ Bk^T - 2µ(PkB)^T + 2 Bk^T W = 0
                # ((ADk)^T (ADk)k + µI) Bk^T + Bk^T W = (ADk)^T Xk + µ(PkB)^T
                # Sylvester equation: AX + XB = Q
                # X = Bk^T, A = ((ADk)^T (ADk)k + µI), B = L, Q = (ADk)^T Xk + µ(PkB)^T
                A = ADk.T @ ADk + self.mu[k] * np.eye(ADk.shape[1])
                B = self.l2_penalty
                Q = ADk.T @ self.X[k] + self.mu[k] * PkB.T
                decomposition.B[k][:] = sla.solve_sylvester(A, B, Q).T
            else:
                # || (ADk) B^T - Xk ||^2 + µ ||Bk^T - (PkB)^T||^2
                # B (ADk)^T (ADk) B^T - 2 B (ADk)^T Xk + µ Bk Bk^T - 2µ Bk (PkB)^T
                # Differentiate
                # 2 (ADk)^T (ADk) B^T - 2 (ADk)^T Xk + 2µ Bk^T - 2µ (PkB)^T = 0
                # 2 (ADk)^T (ADk) B^T  + 2µ Bk^T =  2 (ADk)^T Xk + 2µ (PkB)^T 
                # ((ADk)^T (ADk) + µI) B^T = (ADk)^T Xk + µ (PkB)^T 

                I = np.eye(ADk.shape[1])
                lhs = ADk.T@ADk + self.mu[k]*I
                rhs = (ADk.T) @ self.X[k] + self.mu[k] * (PkB.T)
                decomposition.B[k][:] = np.linalg.solve(lhs, rhs).T
        
    def get_coupling_errors(self, decomposition):
        coupling_error = 0
        for k in range(self.K):
            PkB = self.projection_matrices[k]@self.blueprint_B
            coupling_error += np.linalg.norm(PkB - decomposition.B[k])/np.linalg.norm(decomposition.B[k])
        return coupling_error
    
    def get_weighted_squared_coupling_errors(self, decomposition):
        err = 0
        for k in range(self.K):
            PkB = self.projection_matrices[k]@self.blueprint_B
            err += self.mu[k] * np.linalg.norm(PkB - decomposition.B[k])**2
        return err

# Main class
class BCDCoupledMatrixDecomposer(BaseDecomposer):
    """
    Example usage

    .. code::

        sub_problems = [...]
        pf2 = BCDCoupledMatrixDecomposer(rank, subproblems)
        pf2.fit(X)
    

    Behind the scenes, the fit method calls

    .. code::
        pf2._init_fit(...)
            pf2._init_decomposition(...)
            for subproblem in pf2.sub_problems:
                sub_problem.init_subproblem(...)
            # Prepare for convergence checking

        pf2._fit(...)
            pf2._update_decomposition(...)
                for sub_problem in pf2.sub_problems:
                    sub_problem.update_decomposition(...)
            pf2._after_fit_iteration(..)
    
    Arguments
    ---------
    rank : int
        Rank of the decomposition
    sub_problems : list
        List of length 3, each element is a subproblem instance
        that takes care of updating the components for one mode
    max_its : int
        Maximum number of iterations
    convergence_tol : float
        Relative convergence tolerance
    init: str
        Method used to initialise the components random or parafac2_als
    loggers : list
        List of loggers, alternatively None if no loggers are used
    checkpoint_frequency : int 
        How often checkpoints should be stored to disk
    checkpoint_path : str or Path
        Path to HDF5 file the checkpoints should be stored in
    absolute_tol : float
        Absolute tolerance. Iterations stop if loss decrease is less than this
    problem_order : tuple
        Tuple containing the numbers 0, 1 and 2. 
        Specifies the order in which modes are updated
    convergence_method : str
        Method used to specify convergence, admm or flex
    """
    DecompositionType = tenkit.decomposition.CoupledMatrices
    def __init__(
        self,
        rank,
        sub_problems,
        max_its=1000,
        convergence_tol=1e-6,
        init='random',
        loggers=None,
        checkpoint_frequency=None,
        checkpoint_path=None,
        print_frequency=None,
        convergence_check_frequency=1,
        absolute_tol=1e-16,
        problem_order=(1, 0, 2),
        convergence_method="admm",
        store_first_checkpoint=False,
        init_params=None
    ):
        super().__init__(
            max_its=max_its,
            convergence_tol=convergence_tol,
            loggers=loggers,
            checkpoint_frequency=checkpoint_frequency,
            checkpoint_path=checkpoint_path,
            print_frequency=print_frequency,
        )
        self.rank = rank
        self.sub_problems = sub_problems
        self.init = init
        self.convergence_check_frequency = convergence_check_frequency
        self.auxiliary_variables = [{}, {}, {}]
        self.absolute_tolerance = absolute_tol
        self.problem_order = problem_order
        self.convergence_method = convergence_method
        self.store_first_checkpoint = store_first_checkpoint
        self._cache = {}
        if init_params is None:
            init_params = {}
        self.init_params = init_params

    @property
    def regularisation_penalty(self):
        return sum(sp.regulariser(self.decomposition) for sp in self.sub_problems)

    @property
    def loss(self):
        return self.SSE + self.regularisation_penalty

    def _fit(self):
        if self.checkpoint_frequency > 0 and self.store_first_checkpoint:
            # TODO: Consider to do this for tenkit too
            self.store_checkpoint()


        for it in range(self.max_its - self.current_iteration):
            if self._has_converged():
                break
            if it == 0 and self.store_first_checkpoint:
                for logger in self.loggers:
                    logger.log(self)

            self._update_decomposition()
            self._after_fit_iteration()

            if self.current_iteration % self.print_frequency == 0 and self.print_frequency > 0:
                rel_change = self._rel_function_change
                ce_string = ', '.join(f'{ce:.1g}' for ce in self.coupling_errors)
                print(f'{self.current_iteration:6d}: The MSE is {self.MSE:.4g}, f is {self.loss:4g}, '
                      f'improvement is {rel_change:.2g}, coupling errors: {ce_string}')

        if (
            ((self.current_iteration) % self.checkpoint_frequency != 0) and 
            (self.checkpoint_frequency > 0)
        ):
            self.store_checkpoint() 
        
    def _update_decomposition(self):
        for problem_id in self.problem_order:
        # The function below updates the decomposition and the projected X inplace.
        # print(f'Before {self.current_iteration:6d}B: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
        #               f'improvement is {self._rel_function_change:g}')
            self.sub_problems[problem_id].update_decomposition(
                self.decomposition, self.auxiliary_variables,
            )
            self._cache = {}

    def _check_valid_components(self, decomposition):
        assert type(decomposition) == tenkit.decomposition.EvolvingTensor

    def init_pf2_als(self):
        # Should we have an init_params dict?
        init_params = self.init_params
        init_params["max_its"] = init_params.get("max_its", 1)
        init_params["non_negativity_constraints"] = init_params.get("non_negativity_constraints", [False, False, True])
        init_params["print_frequency"] = init_params.get("print_frequency", -1)
        pf2 = tenkit.decomposition.Parafac2_ALS(
            self.rank,
            **init_params
        )
        pf2.fit(list(self.X))
        A = pf2.decomposition.A
        B = pf2.decomposition.B
        C = pf2.decomposition.C

        norm_A = np.linalg.norm(A, axis=0, keepdims=True)
        norm_B = np.linalg.norm(B[0], axis=0, keepdims=True)
        norm_C = np.linalg.norm(C, axis=0, keepdims=True)
        norm = (norm_A*norm_B*norm_C)**(1/3)
        A = norm * A / norm_A
        B = [norm * Bk / norm_B for Bk in B]
        C = norm * C / norm_C

        self.decomposition = self.DecompositionType(
            A, B, C
        )
        self.auxiliary_variables[1]['projection_matrices'] = list(pf2.decomposition.projection_matrices)
        self.auxiliary_variables[1]['blueprint_B'] = norm * pf2.decomposition.blueprint_B / norm_B

    def init_components(self, initial_decomposition=None):
        if self.init.lower() == 'random':
            self.init_random()
        elif self.init.lower() == 'parafac2_als':
            self.init_pf2_als()
        elif self.init.lower() == 'from_checkpoint':
            self.load_checkpoint(initial_decomposition)
        elif self.init.lower() == 'precomputed':
            self._check_valid_components(initial_decomposition)
            self.decomposition = initial_decomposition
        elif Path(self.init).is_file():
            self.load_checkpoint(self.init)
        else:
            # TODO: better message
            raise ValueError('Init method must be either `random`, `cp`, `parafac2`, `svd`, `from_checkpoint`, `precomputed` or a path to a checkpoint.')

    def init_random(self):
        self.decomposition = self.DecompositionType.random_init(self.X_shape, rank=self.rank)
        self.decomposition.A[:] = getattr(np.random, INIT_METHOD_A)(size=self.decomposition.A.shape)
        for Bi in self.decomposition.B:
            Bi[:] = getattr(np.random, INIT_METHOD_B)(size=Bi.shape)
        self.decomposition.C[:] = getattr(np.random, INIT_METHOD_C)(size=self.decomposition.C.shape)
    
    def _init_fit(self, X, max_its, initial_decomposition):
        super()._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition)

        for i, sub_problem in enumerate(self.sub_problems):
            sub_problem.init_subproblem(i, self)
        self.prev_loss = self.loss
        self.prev_sse = self.SSE
        self.prev_reg = self.regularisation_penalty
        self.prev_coupling_errors = self.coupling_errors
        self._rel_function_change = np.inf
    
    @property
    def reconstructed_X(self):
        return self.decomposition.construct_slices()

    @property
    def coupling_errors(self):
        coupling_errors = []
        for sub_problem in self.sub_problems:
            if hasattr(sub_problem, 'get_coupling_errors'):
                coupling_errors += sub_problem.get_coupling_errors(self.decomposition)
        return coupling_errors

    def set_target(self, X):
        if not isinstance(X, list):
            self.target_tensor = X
            X = np.ascontiguousarray(X.transpose(2, 0, 1))
        
        self.X = X
        self.X_shape = [len(X[0]), [Xk.shape[1] for Xk in X], len(X)]    # len(A), len(Bk), len(C)
        self.X_norm = np.sqrt(sum(np.linalg.norm(Xk)**2 for Xk in X))
        self.num_X_elements = sum([np.prod(s) for s in self.X_shape])

    @property
    def SSE(self):
        return utils.slice_SSE(self.X, self.reconstructed_X)

        #if 'sse' in self._cache:
        #    return self._cache['sse']
        #self._cache['sse'] = utils.slice_SSE(self.X, self.reconstructed_X)
        #return self._cache['sse']

    @property
    def MSE(self):
        return self.SSE/self.decomposition.num_elements

    def checkpoint_callback(self):
        extra_params = {}
        for i, auxiliary_vars in enumerate(self.auxiliary_variables):
            auxiliary_vars = {f"mode_{i}_{key}": value for key, value in auxiliary_vars.items()}
            extra_params = {**extra_params, **auxiliary_vars}
        
        return extra_params
    
    def load_checkpoint(self, checkpoint_path, load_it=None):
        """Load the specified checkpoint at the given iteration.

        If ``load_it=None``, then the latest checkpoint will be used.
        """
        super().load_checkpoint(checkpoint_path, load_it=load_it)
        with h5py.File(checkpoint_path, "r") as h5:
            if 'final_iteration' not in h5.attrs:
                raise ValueError(f'There is no checkpoints in {checkpoint_path}')

            if load_it is None:
                load_it = h5.attrs['final_iteration']
            self.current_iteration = load_it

            group_name = f'checkpoint_{load_it:05d}'
            if group_name not in h5:
                raise ValueError(f'There is no checkpoint {group_name} in {checkpoint_path}')

            checkpoint_group = h5[f'checkpoint_{load_it:05d}']
            # TODO: clean this up
            try:
                initial_decomposition = self.DecompositionType.load_from_hdf5_group(checkpoint_group)
                for sub_problem in self.sub_problems:
                    sub_problem.load_from_hdf5_group(checkpoint_group)
            except KeyError:
                warn("Crashed at final iteration, loading previous iteration")
                groups = [g for g in h5 if "checkpoint_" in g]
                groups.sort()
                for group in sorted(groups, reverse=True):
                    try:
                        initial_decomposition = self.DecompositionType.load_from_hdf5_group(h5[group])
                        for sub_problem in self.sub_problems:
                            sub_problem.load_from_hdf5_group(group)
                    except KeyError:
                        pass
                    else:
                        break
                else:
                    raise ValueError("No valid decomposition")

    def _has_converged(self):
        if self.convergence_method == "flex":
            if self.current_iteration % (self.max_its / 10) != 0:
                return False
            coupling_error = self.sub_problems[1].get_weighted_squared_coupling_errors(self.decomposition)
            sse = self.SSE
            if not hasattr(self, "conv_criterion"):
                self.conv_criterion = (coupling_error + sse) / self.X_norm
                return False
            prev_conv_crit = self.conv_criterion
            self.conv_criterion = (coupling_error + sse) / self.X_norm
            self._rel_function_change = self.conv_criterion
            return np.abs(prev_conv_crit - self.conv_criterion)/prev_conv_crit < self.convergence_tol

        elif self.convergence_method == "admm":
            if self.current_iteration == 0:
                return False
            loss = self.loss
            self._rel_function_change = abs(self.prev_loss - self.loss) / self.prev_loss
            self.prev_loss = loss

            coupling_errors = self.coupling_errors
            for coupling_error, prev_coupling_error in zip(coupling_errors, self.prev_coupling_errors):
                coupling_change = abs(prev_coupling_error - coupling_error)
                relative_coupling_change = coupling_change/prev_coupling_error
                if coupling_error > self.absolute_tolerance and relative_coupling_change > self.convergence_tol:
                    return False
            self.prev_coupling_errors = coupling_errors
            
            return loss < self.absolute_tolerance or self._rel_function_change < self.convergence_tol
        else:
            raise ValueError("convergence method must be admm or flex")

    @property
    def coupling_error(self):
        return sum(self.coupling_errors)                                                                                              


# TODO: UNUSED IDEAS:
class ConvergenceChecker:
    @abstractmethod
    def check_tolerance(self, decomposer):
        pass
    
class LossChecker:
    def __init__(self, relative_tolerance, absolute_tolerance):
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
        self.previous_loss = None
    
    def check_tolerance(self, decomposer):
        if self.previous_loss is None:
            self.previous_loss = decomposer.loss
            return False
        
        # Shorter names to get equations on a single line
        loss = decomposer.loss
        prev_loss = self.previous_loss
        abs_tol = self.absolute_tolerance
        rel_tol = self.relative_tolerance
        rel_criterion = prev_loss - loss / rel_tol*prev_loss
        abs_criterion = prev_loss - loss < abs_tol
        
        self.previous_loss = loss
        return rel_criterion or abs_criterion

# Alias to ensure that old code works:
BlockEvolvingTensor = BCDCoupledMatrixDecomposer

from copy import copy
from pathlib import Path
import h5py
from textwrap import dedent
from warnings import warn
    
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla
import scipy.sparse as sparse
import pyamg

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from tenkit import base
from tenkit.decomposition.cp import get_sse_lhs
from tenkit.decomposition.parafac2 import BaseParafac2, compute_projected_X, Parafac2_ALS
from tenkit.decomposition.decompositions import KruskalTensor, Parafac2Tensor
from tenkit.decomposition.base_decomposer import BaseDecomposer
from tenkit.decomposition import decompositions
from tenkit import utils
from tenkit.decomposition.utils import quadratic_form_trace
from . import _tv_prox


# Default callback
def noop(*args, **kwargs):
    pass


class BaseSubProblem:
    def __init__(self):
        pass

    def update_decomposition(self, X, decomposition):
        pass

    def regulariser(self, factor) -> float:
        return 0
    
    @property
    def checkpoint_params(self):
        return {}

    def load_from_hdf5_group(self, group):
        pass


class ADMM(BaseSubProblem):
    def __init__(self, mode, ridge_penalty=0, non_negativity=False, max_its=10, tol=1e-5, rho=None):
        self.tol = tol
        self.ridge_penalty = ridge_penalty
        self.non_negativity = non_negativity
        self.mode = mode
        self._matrix_khatri_rao_product_cache = None
        self.initialised = False
        self.aux_factor_matrix = None
        self.dual_variables = None
        self.cache = None
        self.rho = rho
        self.auto_rho = (rho is None)
        self.max_its = max_its
    
    def initialise(self, decomposition):
        shape = decomposition.factor_matrices[self.mode].shape
        self.aux_factor_matrix = np.random.uniform(size=shape)
        self.dual_variables = np.random.uniform(size=shape)
        self.initialised = True


    def update_decomposition(self, X, decomposition):
        if not self.initialised:
            self.initialise(decomposition)
        self.cache = {}
        self._recompute_normal_equation(X, decomposition)
        self._set_rho(decomposition)
        self._recompute_cholesky_cache(decomposition)

        for i in range(self.max_its):
            self._update_factor(decomposition)
            self._update_aux_factor_matrix(decomposition)
            self._update_duals(decomposition)

            if self._has_converged(decomposition):
                return
    
    def _recompute_normal_equation(self, X, decomposition):
        self.cache["normal_eq_lhs"] = get_sse_lhs(decomposition.factor_matrices, self.mode)
        self.cache["normal_eq_rhs"] = base.matrix_khatri_rao_product(X, decomposition.factor_matrices, self.mode)


    def _set_rho(self, decomposition):
        if not self.auto_rho:
            return
        else:
            normal_eq_lhs = self.cache['normal_eq_rhs']
            self.rho = np.trace(normal_eq_lhs)/decomposition.rank

    def _recompute_cholesky_cache(self, decomposition):
        # Prepare to compute choleskys
        lhs = self.cache['normal_eq_lhs']
        I = np.eye(decomposition.rank)
        self.cache['cholesky'] = sla.cho_factor(lhs + (0.5*self.rho)*I)

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
            self.aux_factor_matrix =  np.maximum(perturbed_factor, 0)
        else:
            self.aux_factor_matrix = perturbed_factor
        pass
    
    def _has_converged(self, decomposition):

        factor_matrix = decomposition.factor_matrices[self.mode]
        coupling_error = np.linalg.norm(factor_matrix-self.aux_factor_matrix)**2
        coupling_error /= np.linalg.norm(self.aux_factor_matrix)**2
        
        aux_change_sq = np.linalg.norm(self.aux_factor_matrix-self.previous_factor_matrix)**2
        
        dual_var_norm_sq = np.linalg.norm(self.dual_variables)**2
        aux_change_criterion = (aux_change_sq + 1e-16) / (dual_var_norm_sq + 1e-16)
        if self.verbose:
            print("primal criteria", coupling_error, "dual criteria", aux_change_sq)
        return coupling_error < self.tol and aux_change_criterion < self.tol
class NotUpdating(BaseSubProblem):
    def update_decomposition(self, X, decomposition):
        pass


class RLS(BaseSubProblem):
    def __init__(self, mode, ridge_penalty=0, non_negativity=False):
        self.ridge_penalty = ridge_penalty
        self.non_negativity = non_negativity
        self.mode = mode
        self._matrix_khatri_rao_product_cache = None
    
    def update_decomposition(self, X, decomposition):
        lhs = get_sse_lhs(decomposition.factor_matrices, self.mode)
        rhs = base.matrix_khatri_rao_product(X, decomposition.factor_matrices, self.mode)

        self._matrix_khatri_rao_product_cache = rhs

        rightsolve = self._get_rightsolve()

        decomposition.factor_matrices[self.mode][:] = rightsolve(lhs, rhs)
    
    def _get_rightsolve(self):
        rightsolve = base.rightsolve
        if self.non_negativity:
            rightsolve = base.non_negative_rightsolve
        if self.ridge_penalty:
            rightsolve = base.add_rightsolve_ridge(rightsolve, self.ridge_penalty)
        return rightsolve


class BaseParafac2SubProblem(BaseSubProblem):
    _is_pf2_evolving_mode = True
    mode = 1
    def __init__(self):
        pass

    def update_decomposition(self, X, decomposition, projected_X=None, should_update_projections=True):
        pass


class Parafac2RLS(BaseParafac2SubProblem):
    SKIP_CACHE = False
    def __init__(self, ridge_penalty=0):
        self.ridge_penalty = ridge_penalty
        self.non_negativity = False
        self._callback = noop

    def compute_projected_X(self, projection_matrices, X, out=None):
        return compute_projected_X(projection_matrices, X, out=out)

    def update_projections(self, X, decomposition):
        K = len(X)

        for k in range(K):
            A = decomposition.A
            C = decomposition.C
            blueprint_B = decomposition.blueprint_B

            decomposition.projection_matrices[k][:] = base.orthogonal_solve(
                (C[k]*A)@blueprint_B.T,
                X[k]
            ).T

    def update_decomposition(
        self, X, decomposition, projected_X=None, should_update_projections=True
    ):
        """Updates the decomposition inplace

        If a projected data tensor is supplied, then it is updated inplace
        """
        if should_update_projections:
            self.update_projections(X, decomposition)
            projected_X = self.compute_projected_X(decomposition.projection_matrices, X, out=projected_X)
        
        if projected_X is None:
            projected_X = self.compute_projected_X(decomposition.projection_matrices, X, out=projected_X)
        ktensor = KruskalTensor([decomposition.A, decomposition.blueprint_B, decomposition.C])
        RLS.update_decomposition(self, X=projected_X, decomposition=ktensor)

    def _get_rightsolve(self):
        return RLS._get_rightsolve(self)
    
    def regulariser(self, factor):
        if not self.ridge_penalty:
            return 0
        else:
            return self.ridge_penalty*np.sum(factor.ravel()**2)


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


def evolving_factor_total_variation(factor):
    return _tv_prox.TotalVariation(factor, 1).center_penalty()


def total_variation_prox(factor, strength):
    return _tv_prox.TotalVariation(factor, strength).prox()


# TODO: Prox class for each kind of constraint, that will significantly reduce no. arguments
class Parafac2ADMM(BaseParafac2SubProblem):
    # TODO: docstring
    """
    To add new regularisers:
        * Change __init__
        * Change constraint_prox
        * Change regulariser
    """
    # In our notes: U -> dual variable
    #               \tilde{B} -> aux_factor_matrices
    #               B -> decomposition
    SKIP_CACHE = False
    def __init__(
        self,
        rho=None,
        tol=1e-3,
        max_it=5,

        non_negativity=False,
        l2_similarity=None,
        l1_penalty=None,
        tv_penalty=None,
        verbose=False,
        cache_components=True,
        l2_solve_method=None,

        normalize_aux=False,
        normalize_other_modes=False,
        aux_init="same",  # same or random
        dual_init="zeros",  # zeros or random
    ):
        if rho is None:
            self.auto_rho = True
        else:
            self.auto_rho = False

        self.rho = rho
        self.tol = tol
        self.max_it = max_it
     
        self.non_negativity = non_negativity
        self.l2_similarity = l2_similarity
        self.l1_penalty = l1_penalty
        self.tv_penalty = tv_penalty

        self.l2_solve_method = l2_solve_method

        if non_negativity and l2_similarity is not None:
            raise ValueError("Not implemented non negative similarity")
        if l2_similarity is not None and l1_penalty:
            raise ValueError("Not implemented L1+L2 with similarity")
        
        self.verbose = verbose
        self._qr_cache = None
        self._reg_solver = None
        self.dual_variables = None
        self.aux_factor_matrices = None
        self.prev_aux_factor_matrices = None
        self.it_num = 0
        self.normalize_aux = normalize_aux
        self.normalize_other_modes = normalize_other_modes

        self.aux_init = aux_init
        self.dual_init = dual_init

        self._cache_components = cache_components
        self._callback = noop

    def callback(self, X, decomposition, init=False):
        """Calls self._callback, which should have the following signature:
        self._callback(self, X, decomposition, fm, aux_fm, dual_variable)

        By default callback does nothing.
        """
        self._callback(self, X, decomposition, self.aux_factor_matrices, self.dual_variables, init)
    
    def update_decomposition(
        self, X, decomposition, projected_X=None, should_update_projections=True
    ):
        # Normalising other modes must happen before auto rho computation
        if self.normalize_other_modes:
            decomposition.C[...] /= np.linalg.norm(decomposition.C, axis=0, keepdims=True)
            decomposition.A[...] /= np.linalg.norm(decomposition.A, axis=0, keepdims=True)

        self.prepare_for_update(decomposition)

        # Init constraint
        if self.aux_factor_matrices is None or self.it_num == 1 or (not self._cache_components):
            self.aux_factor_matrices = self.init_constraint(decomposition)
        
        if self.normalize_aux:
            self.aux_factor_matrices = [aux_fm/np.linalg.norm(aux_fm, axis=0, keepdims=True) for aux_fm in self.aux_factor_matrices]

        # Init dual variables
        if self.dual_variables is None or self.it_num == 1 or (not self._cache_components):
            self.dual_variables = self.init_duals(decomposition)

        if projected_X is None:
            projected_X = self.compute_projected_X(decomposition.projection_matrices, X, out=projected_X)

        # The decomposition is modified inplace each iteration
        for i in range(self.max_it):
            self.callback(X, decomposition, init=(i==0))
            self.update_blueprint(projected_X, decomposition)

            if should_update_projections:
                self.update_projections(X, decomposition)
                projected_X = self.compute_projected_X(decomposition.projection_matrices, X, out=projected_X)

            self.update_blueprint(projected_X, decomposition)

            self.aux_factor_matrices = self.compute_next_aux_factor_matrices(decomposition)
            self.update_dual(decomposition)

            if self.has_converged(decomposition):
                break
        
        self.callback(X, decomposition, init=False)
        self.it_num += 1

    def prepare_for_update(self, decomposition):
        # Compute LHS of matrix decomposition problem
        lhs = base.khatri_rao(
            decomposition.A, decomposition.C,
        )
        # Calculate rho if it is not supplied to __init__
        if self.auto_rho:
            self.rho = np.linalg.norm(lhs)**2/decomposition.rank
        # Proximal term too keep update close to auxillary matrices
        reg_lhs = np.vstack([np.identity(decomposition.rank) for _ in decomposition.B])
        reg_lhs *= np.sqrt(self.rho/2)
        # Combine to one system
        lhs = np.vstack([lhs, reg_lhs])
        # Store QR factorisation for efficiency
        self._qr_cache = np.linalg.qr(lhs)

        # If l2 similarity is used, prepare solver instance
        if self.l2_similarity is not None:
            if sparse.issparse(self.l2_similarity):
                I = sparse.eye(self.l2_similarity.shape[0])
            else:
                I = np.identity(self.l2_similarity.shape[0])
            reg_matrix = self.l2_similarity + 0.5*self.rho*I

            self._reg_solver = _SmartSymmetricPDSolver(reg_matrix, method=self.l2_solve_method)

    def init_constraint(self, decomposition):
        if self.aux_init == "same":
            init_P, init_B = decomposition.projection_matrices, decomposition.blueprint_B
            B = [P_k@init_B for P_k in init_P]
            return [
                self.constraint_prox(B_k, decomposition) for k, B_k in enumerate(B)
            ]
        elif self.aux_init == "random":
            B = [np.random.standard_normal(Bk.shape)*np.median(np.abs(Bk)) for Bk in decomposition.B]
            return [self.constraint_prox(Bk, decomposition) for Bk in B]
        else:
            raise ValueError(f"Invalid aux init, {self.aux_init}")

    def init_duals(self, decomposition):
        if self.dual_init == "zeros":
            return [np.zeros_like(Bk) for Bk in decomposition.B]
        elif self.dual_init == "random":
            return [np.random.standard_normal(Bk.shape)*np.median(np.abs(Bk)) for Bk in decomposition.B]
        else:
            raise ValueError(f"Invalid dual init: {self.dual_init}")

    def compute_next_aux_factor_matrices(self, decomposition):
        projections = decomposition.projection_matrices
        blueprint_B = decomposition.blueprint_B
        rank = blueprint_B.shape[1]
        return [
            self.constraint_prox(
                P_k@blueprint_B + self.dual_variables[k], decomposition,
            ) for k, P_k in enumerate(projections)
        ]
        # Below, there is some code that will never be run. I tried introducing it and got some bugs,
        # but I'm uncertain if it was this code or some other change I made. I therefore kept it
        # planning to use it again later.
        # TODO: Can we do this? Is it faster?
        Bks = [P_k@blueprint_B for P_k in projections]
        Bks = np.concatenate(Bks, axis=1)
        dual_variables = np.concatenate([dual_variable for dual_variable in self.dual_variables], axis=1)

        proxed = self.constraint_prox(Bks + dual_variables, decomposition)
        return [
            proxed[:, k*rank:(k+1)*rank] for k, _ in enumerate(projections)
        ]

    def constraint_prox(self, x, decomposition):
        # TODO: Comments
        # TODO: Check compatibility between different proxes
        # The above todo is not necessary if we implement a separate prox-class.
        if self.non_negativity and self.l1_penalty:
            return np.maximum(x - 2*self.l1_penalty/self.rho, 0)
        elif self.tv_penalty:
            return total_variation_prox(x, 2*self.tv_penalty/self.rho)
        elif self.non_negativity:
            return np.maximum(x, 0)      
        elif self.l1_penalty:
            return np.sign(x)*np.maximum(np.abs(x) - 2*self.l1_penalty/self.rho, 0)
        elif self.l2_similarity is not None:
            rhs = 0.5*self.rho*x
            return self._reg_solver.solve(rhs)
        else:
            return x

    def update_dual(self, decomposition):
        # TODO: Kommentarer
        for P_k, aux_fm, dual_variable in zip(decomposition.projection_matrices, self.aux_factor_matrices, self.dual_variables):
            B_k = P_k@decomposition.blueprint_B
            dual_variable += B_k - aux_fm

    def update_projections(self, X, decomposition):
        # Triangle equation from notes
        A = decomposition.A
        blueprint_B = decomposition.blueprint_B
        C = decomposition.C
        for k, X_k in enumerate(X):
            unreg_lhs = (A*C[k])@(blueprint_B.T)
            reg_lhs = np.sqrt(self.rho/2)*(blueprint_B.T)
            lhs = np.vstack((unreg_lhs, reg_lhs))

            unreg_rhs = X_k
            reg_rhs = np.sqrt(self.rho/2)*(self.aux_factor_matrices[k] - self.dual_variables[k]).T
            rhs = np.vstack((unreg_rhs, reg_rhs))
            
            decomposition.projection_matrices[k][:] = base.orthogonal_solve(lhs, rhs).T

    def update_blueprint(self, projected_X, decomposition):
        # Square equation from notes
        Q, R = self._qr_cache
        
        rhs = base.unfold(projected_X, 1).T
        projected_aux = [
            (aux_fm - dual_variable).T@projection
            for aux_fm, dual_variable, projection in zip(
                self.aux_factor_matrices, self.dual_variables, decomposition.projection_matrices
            )
        ]
        reg_rhs = np.vstack(projected_aux)
        reg_rhs *= np.sqrt(self.rho/2)
        rhs = np.vstack([rhs, reg_rhs])

        decomposition.blueprint_B[:] = sla.solve_triangular(R, Q.T@rhs, lower=False).T

    def compute_projected_X(self, projection_matrices, X, out=None):
        return compute_projected_X(projection_matrices, X, out=out)

    def _compute_relative_coupling_error(self, fms, aux_factor_matrices):
        gap_sq = sum(np.linalg.norm(fm - aux_fm)**2 for fm, aux_fm in zip(fms, aux_factor_matrices))
        aux_norm_sq = sum(np.linalg.norm(aux_fm)**2 for aux_fm in aux_factor_matrices)
        return gap_sq/aux_norm_sq

    def has_converged(self, decomposition):
        if self.prev_aux_factor_matrices is None:
            self.prev_aux_factor_matrices = self.aux_factor_matrices
            return False
        coupling_error = self._compute_relative_coupling_error(decomposition.B, self.aux_factor_matrices)
        aux_change_sq = sum(
            np.linalg.norm(aux_fm - prev_aux_fm)**2 for aux_fm, prev_aux_fm in zip(self.aux_factor_matrices, self.prev_aux_factor_matrices)
        )
        dual_var_norm_sq = sum(np.linalg.norm(dual_var)**2 for dual_var in self.dual_variables)
        aux_change_criterion = (aux_change_sq + 1e-16) / (dual_var_norm_sq + 1e-16)
        if self.verbose:
            print("primal criteria", coupling_error, "dual criteria", aux_change_sq)
        
        self.prev_aux_factor_matrices = self.aux_factor_matrices
        return coupling_error < self.tol and aux_change_criterion < self.tol
    
    def regulariser(self, factor_matrices):
        reg = 0
        if self.l2_similarity is not None:
            factor_matrices = np.array(factor_matrices)
            reg += sum(
                quadratic_form_trace(self.l2_similarity, factor_matrix)
                for factor_matrix in factor_matrices
            )
        if self.l1_penalty is not None:
            factor_matrices = np.array(factor_matrices)
            reg += self.l1_penalty*np.linalg.norm(factor_matrices.ravel(), 1)
        if self.tv_penalty is not None:
            factor_matrices = np.array(factor_matrices)
            reg += self.tv_penalty*evolving_factor_total_variation(factor_matrices)
        return reg
    
    @property
    def checkpoint_params(self):
        aux_factor_matrices = {
            f"aux_fm_{i:04d}": aux_fm for i, aux_fm in enumerate(self.aux_factor_matrices)
        }
        duals = {
            f"dual_var_{i:04d}": dual_var for i, dual_var in enumerate(self.dual_variables)
        }
        checkpoint_params = {**aux_factor_matrices, **duals}

        return checkpoint_params

    def load_from_hdf5_group(self, group):
        aux_fm_names = [dataset_name for dataset_name in group if dataset_name.startswith("aux_fm_")]
        aux_fm_names.sort()
        dual_names = [dataset_name for dataset_name in group if dataset_name.startswith("dual_var_")]
        dual_names.sort()

        self.aux_factor_matrices = [group[aux_fm_name][:] for aux_fm_name in aux_fm_names]
        self.dual_var = [group[dual_name][:] for dual_name in dual_names]


class BlockParafac2(BaseDecomposer):
    DecompositionType = decompositions.Parafac2Tensor
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
        projection_update_frequency=5,
        convergence_check_frequency=1,
        normalize_B=False,
        normalize_B_after_update=False
    ):
        # Make sure that the second mode solves for evolving components
        if (
            not hasattr(sub_problems[1], '_is_pf2_evolving_mode') or 
            not sub_problems[1]._is_pf2_evolving_mode
        ):
            raise ValueError(
                'Second sub problem must follow PARAFAC2 constraints. If it does, '
                'ensure that `sub_problem._is_pf2 == True`.'
            )

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
        self.projection_update_frequency = projection_update_frequency
        self.convergence_check_frequency = convergence_check_frequency
        self.normalize_B = normalize_B
        self.normalize_B_after_update = normalize_B_after_update

    def _check_valid_components(self, decomposition):
        return BaseParafac2._check_valid_components(self, decomposition)

    @property
    def regularisation_penalty(self):
        factor_matrices = [
            self.decomposition.A,
            np.array(self.decomposition.B),
            self.decomposition.C
        ]
        return sum(sp.regulariser(fm) for sp, fm in zip(self.sub_problems, factor_matrices))

    @property
    def loss(self):
        return self.SSE + self.regularisation_penalty

    def _update_parafac2_factors(self):
        should_update_projections = self.current_iteration % self.projection_update_frequency == 0
        # The function below updates the decomposition and the projected X inplace.
        # print(f'Before {self.current_iteration:6d}B: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
        #               f'improvement is {self._rel_function_change:g}')
        l = self.loss
        if self.normalize_B:
            norms = np.linalg.norm(self.decomposition.blueprint_B, axis=0, keepdims=True)
            self.decomposition.blueprint_B[:] = self.decomposition.blueprint_B/norms
            self.decomposition.C[:] = self.decomposition.C*norms

        self.sub_problems[1].update_decomposition(
            self.X, self.decomposition, self.projected_X, should_update_projections=should_update_projections
        )

        if self.normalize_B_after_update:
            norms = np.linalg.norm(self.decomposition.blueprint_B, axis=0, keepdims=True)
            self.decomposition.blueprint_B[:] = self.decomposition.blueprint_B/norms
            self.decomposition.C[:] = self.decomposition.C*norms
        # print(f'Before {self.current_iteration:6d}A: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
        #               f'improvement is {self._rel_function_change:g}')
        self.sub_problems[0].update_decomposition(
            self.projected_X, self.cp_decomposition
        )
        # print(f'Before {self.current_iteration:6d}C: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
        #               f'improvement is {self._rel_function_change:g}')
        self.sub_problems[2].update_decomposition(
            self.projected_X, self.cp_decomposition
        )

    def _fit(self):
        for it in range(self.max_its - self.current_iteration):
            if self._has_converged():
                break

            self._update_parafac2_factors()
            self._after_fit_iteration()

            if self.current_iteration % self.print_frequency == 0 and self.print_frequency > 0:
                rel_change = self._rel_function_change

                print(f'{self.current_iteration:6d}: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
                      f'improvement is {rel_change:g}')

        if (
            ((self.current_iteration) % self.checkpoint_frequency != 0) and 
            (self.checkpoint_frequency > 0)
        ):
            self.store_checkpoint()

    def init_components(self, initial_decomposition=None):
        if self.init == 'ALS':
            self.pf2 = Parafac2_ALS(self.rank, max_its=100, print_frequency=-1)
            self.pf2.fit([Xi for Xi in self.X])
            self.decomposition = self.pf2.decomposition
        else:
            BaseParafac2.init_components(self, initial_decomposition=initial_decomposition)

    def _has_converged(self):
        has_converged = False

        should_check_convergence = self.current_iteration % self.convergence_check_frequency == 0
        if should_check_convergence and self.current_iteration > 0:
            sse = self.SSE
            reg = self.regularisation_penalty
            loss = self.loss

            # TODO: Do this check for all modes
            if hasattr(self.sub_problems[1], "_compute_relative_coupling_error"):
                rel_coupling = self.sub_problems[1]._compute_relative_coupling_error(
                    self.decomposition.B,
                    self.sub_problems[1].aux_factor_matrices
                )
            else:
                rel_coupling = 0

            is_first_it = (self.prev_sse is None)
            # Used for printing
            self._rel_function_change = (self.prev_loss - loss)/self.prev_loss

            if is_first_it:
                return False

            # TODO: Double check coupling error convergence
            # TODO: See if reg convergence should be checked for each mode separately
            # Convergence if 
            # (prev - curr)/prev < tol
            # prev - curr < prev*tol
            # prev - prev*tol - curr < 0
            # (1 - tol)prev - curr < 0
            # (1 - tol)prev < curr
            tol = (1 - self.convergence_tol)**self.convergence_check_frequency
            has_converged = (sse >= tol*self.prev_sse) and (reg >= tol*self.prev_reg)
            has_converged = has_converged and (rel_coupling >= tol*self.prev_rel_coupling)
            self.prev_rel_coupling = rel_coupling
            self.prev_loss = loss
            self.prev_reg = reg
            self.prev_sse = sse
            self.prev_rel_coupling = rel_coupling

        return has_converged

    def _init_fit(self, X, max_its, initial_decomposition):
        super()._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition)
        self.cp_decomposition = KruskalTensor(
            [self.decomposition.A, self.decomposition.blueprint_B, self.decomposition.C]
        )
        self.projected_X = compute_projected_X(self.decomposition.projection_matrices, self.X)
        self.prev_loss = self.loss
        self.prev_sse = self.SSE
        self.prev_reg = self.regularisation_penalty
        self.prev_rel_coupling = None
        self._rel_function_change = np.inf
    
    def init_random(self):
        return BaseParafac2.init_random(self)
    
    def init_svd(self):
        return BaseParafac2.init_svd(self)

    def init_cp(self):
        return BaseParafac2.init_cp(self)

    @property
    def reconstructed_X(self):
        return self.decomposition.construct_slices()
    
    def set_target(self, X):
        BaseParafac2.set_target(self, X)

    @property
    def SSE(self):
        return utils.slice_SSE(self.X, self.reconstructed_X)

    @property
    def MSE(self):
        return self.SSE/self.decomposition.num_elements
    
    def checkpoint_callback(self):
        extra_params = {}
        for sub_problem in self.sub_problems:
            extra_params = {**extra_params, **sub_problem.checkpoint_params}
        
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
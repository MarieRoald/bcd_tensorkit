from abc import ABC, abstractproperty, abstractmethod
import tenkit
from tenkit.decomposition import BaseDecomposer
from pathlib import Path
import numpy as np
import scipy.linalg as sla
from tenkit import base
import h5py
from warnings import warn
from tenkit import utils

from tenkit.decomposition.cp import get_sse_lhs

class BaseSubProblem(ABC):
    def __init__(self):
        pass

    def init_subproblem(self, mode, auxiliary_variables):
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


class Mode0RLS(BaseSubProblem):
    def __init__(self, non_negativity=False, ridge_penalty=0):
        self.non_negativity = non_negativity
        self.ridge_penalty = ridge_penalty

    def init_subproblem(self, mode, auxiliary_variables, rank, X):
        assert mode == 0
        self.mode = mode
        self.X = X
        self.unfolded_X = np.concatenate([X_slice for X_slice in self.X], axis=1)

    def _get_rightsolve(self):
        rightsolve = base.rightsolve
        if self.non_negativity:
            rightsolve = base.non_negative_rightsolve
        if self.ridge_penalty:
            rightsolve = base.add_rightsolve_ridge(rightsolve, self.ridge_penalty)
        return rightsolve
        
    def update_decomposition(self, decomposition, auxiliary_variables):
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
    def __init__(self, non_negativity=False, max_its=10, tol=1e-5, rho=None, verbose=False):
        self.tol = tol
        self.non_negativity = non_negativity
        self.max_its = max_its
        self.rho = rho 
        self.auto_rho = (rho is None)
        self.verbose = verbose
    
    def init_subproblem(self, mode, auxiliary_variables, rank, X):
        assert mode == 0
        self.mode = mode
        self.X = X
        self.unfolded_X = np.concatenate([X_slice for X_slice in self.X], axis=1)

        I = X[0].shape[0]
        self.aux_factor_matrix = np.random.uniform(0, 1, size=(I, rank))
        self.dual_variables = np.random.uniform(0, 1, size=(I, rank))
        auxiliary_variables[0]['aux_factor_matrix'] = self.aux_factor_matrix
        auxiliary_variables[0]['dual_variables'] = self.dual_variables

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
                return
    
    def regulariser(self, decomposition):
        return 0

    def get_coupling_errors(self, decomposition):
        return [np.linalg.norm(decomposition.A - self.aux_factor_matrix)**2]

    
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



class Mode2RLS(BaseSubProblem):
    def __init__(self, non_negativity=False, ridge_penalty=0):
        self.non_negativity = non_negativity
        self.ridge_penalty = ridge_penalty

    def init_subproblem(self, mode, auxiliary_variables, rank, X):
        assert mode == 2
        self.mode = mode
        self.X = X

    def _get_rightsolve(self):
        rightsolve = base.rightsolve
        if self.non_negativity:
            rightsolve = base.non_negative_rightsolve
        if self.ridge_penalty:
            rightsolve = base.add_rightsolve_ridge(rightsolve, self.ridge_penalty)
        return rightsolve
        
    def update_decomposition(self, decomposition, auxiliary_variables):
        rightsolve = self._get_rightsolve()
        for k, (c_row, factor_matrix) in enumerate(zip(decomposition.C,  decomposition.B)):
            X_k_vec = self.X[k].reshape(-1, 1)
            lhs = base.khatri_rao(decomposition.A, factor_matrix)
            c_row[...] = rightsolve(lhs.T, X_k_vec.T.ravel())

    def regulariser(self, decomposition):
        if not self.ridge_penalty:
            return 0
        return self.ridge_penalty*np.sum(decomposition.B**2)

    def get_coupling_errors(self, decomposition):
        return []


class Mode2ADMM(BaseSubProblem):
    def __init__(self, non_negativity=False, max_its=10, tol=1e-5, rho=None, verbose=False):
        self.tol = tol
        self.non_negativity = non_negativity
        self.max_its = max_its
        self.rho = rho 
        self.auto_rho = (rho is None)
        self.verbose = verbose
    
    def init_subproblem(self, mode, auxiliary_variables, rank, X):
        assert mode == 2
        self.mode = mode
        self.X = X
        self.unfolded_X = np.concatenate([X_slice for X_slice in self.X], axis=1)

        K = len(X)
        self.aux_factor_matrix = np.random.uniform(0, 1, size=(K, rank))
        self.dual_variables = np.random.uniform(0, 1, size=(K, rank))
        auxiliary_variables[self.mode]['aux_factor_matrix'] = self.aux_factor_matrix
        auxiliary_variables[self.mode]['dual_variables'] = self.dual_variables
        
    def update_decomposition(self, decomposition, auxiliary_variables):
        rightsolve = self._get_rightsolve()
        for k, (c_row, factor_matrix) in enumerate(zip(decomposition.C,  decomposition.B)):
            X_k_vec = self.X[k].reshape(-1, 1)
            lhs = base.khatri_rao(decomposition.A, factor_matrix)
            c_row[...] = rightsolve(lhs.T, X_k_vec.T.ravel())

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
                return
    
    def regulariser(self, decomposition):
        return 0

    def get_coupling_errors(self, decomposition):
        return [np.linalg.norm(decomposition.C - self.aux_factor_matrix)**2]

    
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
        self.cache['cholesky'] = [sla.cho_factor(lhs + (0.5*rho)*I) for rho,lhs in zip(self.rho, self.cache['normal_eq_lhs'])]

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




def _random_orthogonal_matrix(num_rows, num_cols):
    return np.linalg.qr(np.random.standard_normal(size=(num_rows, num_cols)))[0]


class DoubleSplittingParafac2ADMM(BaseSubProblem):
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
        l2_solve_method=None,
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

        self._cache = {}        


    def init_subproblem(self, mode, auxiliary_variables, rank, X):
        assert mode == 1

        K = len(X)
        self.mode = mode
        self.X = X

        self._cache['rho'] = [np.inf]*K

        self.blueprint_B = np.random.uniform(size=(rank, rank))
        self.projection_matrices = [np.eye(X[k].shape[1], rank) for k in range(K)]
        self.reg_Bks = [self.prox(np.random.uniform(size=(X[k].shape[1], rank)), k) for k in range(K)]
        self.dual_variables_reg = [np.random.uniform(size=(X[k].shape[1], rank)) for k in range(K)]
        self.dual_variables_pf2 = [np.random.uniform(size=(X[k].shape[1], rank)) for k in range(K)]

        auxiliary_variables[1]['blueprint_B'] = self.blueprint_B
        auxiliary_variables[1]['projection_matrices'] = self.projection_matrices
        auxiliary_variables[1]['reg_Bks'] = self.reg_Bks
        auxiliary_variables[1]['dual_variables_reg'] = self.dual_variables_reg
        auxiliary_variables[1]['dual_variables_pf2'] = self.dual_variables_pf2

    def update_decomposition(
        self, decomposition, auxiliary_variables
    ):
        self._cache = {}
        self.recompute_normal_equation(decomposition)
        self.set_rhos(decomposition)
        self.recompute_cholesky_cache(decomposition)
        # breakpoint()
        # The decomposition is modified inplace each iteration
        for i in range(self.max_it):
            # Update Bks and Pks
            # Update blueprint
            # Update reg matrices
            # Update duals
            # print("unconstrained:", np.linalg.norm(decomposition.B[0], axis=0))
            self.update_unconstrained(decomposition)
            # print("unconstrained:", np.linalg.norm(decomposition.B[0], axis=0))

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
            self._cache['rho'] = [np.trace(lhs)/decomposition.rank for lhs in normal_eq_lhs]

    def recompute_cholesky_cache(self, decomposition):
        # Prepare to compute choleskys
        rhos = self._cache['rho']
        normal_eq_lhs = self._cache['normal_eq_lhs']
        I = np.eye(decomposition.rank)
        self._cache['choleskys'] = [sla.cho_factor(lhs + (rho)*I) for rho, lhs in zip(rhos, normal_eq_lhs)]

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

            prox_rhs = rhs + 0.5*rho*(reg_Bk - dual_variables_reg + P@blueprint_B - dual_variables_pf2)
            decomposition.B[k][...] = sla.cho_solve(chol_lhs, prox_rhs.T).T

    def update_projections(self, decomposition):
        self.previous_projections = [pk for pk in self.projection_matrices]
        K = decomposition.C.shape[0]
        blueprint_B = self.blueprint_B  # DeltaB
        for k in range(K):
            Bk = decomposition.B[k]
            dual_variable_pf2_k = self.dual_variables_pf2[k]
            
            U, s, Vh = np.linalg.svd((Bk + dual_variable_pf2_k)@blueprint_B.T, full_matrices=False)
            self.projection_matrices[k] = U@Vh
    
    def update_blueprint(self, decomposition):
        self.previous_blueprint_B = self.blueprint_B
        K = decomposition.C.shape[0]
        involved_variables = zip(self.projection_matrices, decomposition.B, self.dual_variables_pf2)
        self.blueprint_B[:] = sum(Pk.T@(Bk + dual_pf2_k) for Pk, Bk, dual_pf2_k in involved_variables)/K
    
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
        elif self.tv_penalty:
            raise NotImplementedError
            #return total_variation_prox(factor_matrix, 2*self.tv_penalty/rho)
        elif self.non_negativity:
            return np.maximum(factor_matrix, 0)      
        elif self.l1_penalty:
            return np.sign(factor_matrix)*np.maximum(np.abs(factor_matrix) - 2*self.l1_penalty/rho, 0)
        elif self.l2_similarity is not None:
            raise NotImplementedError
            #rhs = 0.5*rho*factor_matrix
            #return self._reg_solver.solve(rhs)
        else:
            return factor_matrix
    
    def update_duals(self, decomposition):
        K = decomposition.C.shape[0]
        for k in range(K):
            self.dual_variables_reg[k] += decomposition.B[k] - self.reg_Bks[k]
            self.dual_variables_pf2[k] += decomposition.B[k] - self.projection_matrices[k]@self.blueprint_B

    def has_converged(self, decomposition):       
        K = decomposition.C.shape[0]
        
        reg_coupling_error = 0
        pf2_coupling_error = 0
        sum_sq_factor = 0

        relative_reg_change = 0
        sum_sq_dual_reg = 0

        relative_pf2_change = 0
        sum_sq_dual_pf2 = 0

        for k in range(K):
            reg_coupling_error += np.linalg.norm(decomposition.B[k] - self.reg_Bks[k])**2
            pf2_coupling_error += np.linalg.norm(decomposition.B[k] - self.projection_matrices[k]@self.blueprint_B)**2
            sum_sq_factor += np.linalg.norm(decomposition.B[k])

            relative_reg_change += np.linalg.norm(self.previous_reg_Bks - self.reg_Bks[k])**2
            sum_sq_dual_reg += np.linalg.norm(self.dual_variables_reg)**2

            previous_pf2_Bks = self.previous_projections[k]@self.previous_blueprint_B
            pf2_Bk = self.projection_matrices[k]@self.blueprint_B
            relative_pf2_change += np.linalg.norm(previous_pf2_Bks - pf2_Bk)
            sum_sq_dual_pf2 += np.linalg.norm(self.dual_variables_pf2)**2

        relative_reg_change /= (sum_sq_dual_reg + 1e-16)/K
        relative_pf2_change /= (sum_sq_dual_pf2 + 1e-16)/K
        relative_change_criterion = max(relative_pf2_change, relative_reg_change)

        relative_reg_coupling_error = reg_coupling_error/sum_sq_factor/K
        relative_pf2_coupling_error = pf2_coupling_error/sum_sq_factor/K
        relative_coupling_criterion = max(relative_pf2_coupling_error, relative_reg_coupling_error)

        return relative_change_criterion < self.tol and relative_coupling_criterion < self.tol

    def get_coupling_errors(self, decomposition):
        K = len(self.projection_matrices)
        pf2_coupling_error = 0
        reg_coupling_error = 0
        for k in range(K):
            pf2_Bk = self.projection_matrices[k]@self.blueprint_B
            
            pf2_coupling_error += np.linalg.norm(pf2_Bk - decomposition.B[k])**2
            reg_coupling_error += np.linalg.norm(self.reg_Bks[k] - decomposition.B[k])**2

        return reg_coupling_error, pf2_coupling_error


class BlockEvolvingTensor(BaseDecomposer):
    DecompositionType = tenkit.decomposition.EvolvingTensor
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
        absolute_tol=1e-16
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

    def _fit(self):
        for it in range(self.max_its - self.current_iteration):
            if self._has_converged():
                break

            self._update_evolving_tensors_factors()
            self._after_fit_iteration()

            if self.current_iteration % self.print_frequency == 0 and self.print_frequency > 0:
                rel_change = self._rel_function_change

                print(f'{self.current_iteration:6d}: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
                      f'improvement is {rel_change:g}, {self.coupling_errors}')

        if (
            ((self.current_iteration) % self.checkpoint_frequency != 0) and 
            (self.checkpoint_frequency > 0)
        ):
            self.store_checkpoint() 
        
    def _update_evolving_tensors_factors(self):
        # The function below updates the decomposition and the projected X inplace.
        # l = self.loss

        # print(f'Before {self.current_iteration:6d}A: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
        #               f'improvement is {self._rel_function_change:g}')
        self.sub_problems[0].update_decomposition(
            self.decomposition, self.auxiliary_variables,
        )


        # print(f'Before {self.current_iteration:6d}C: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
        #               f'improvement is {self._rel_function_change:g}')
        self.sub_problems[2].update_decomposition(
            self.decomposition, self.auxiliary_variables,
        )

        # print(f'Before {self.current_iteration:6d}B: The MSE is {self.MSE:4g}, f is {self.loss:4g}, '
        #               f'improvement is {self._rel_function_change:g}')
        self.sub_problems[1].update_decomposition(
            self.decomposition, self.auxiliary_variables,
        )

    def _check_valid_components(self, decomposition):
        assert type(decomposition) == tenkit.decomposition.EvolvingTensor

    def init_components(self, initial_decomposition=None):
        if self.init.lower() == 'random':
            self.init_random()
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
    
    def _init_fit(self, X, max_its, initial_decomposition):
        super()._init_fit(X=X, max_its=max_its, initial_decomposition=initial_decomposition)

        for i, sub_problem in enumerate(self.sub_problems):
            sub_problem.init_subproblem(i, self.auxiliary_variables, self.rank, self.X)
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

    def _has_converged(self):
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

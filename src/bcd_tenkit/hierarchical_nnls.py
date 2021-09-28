import numpy as np


def nnls(AtA, AtY, Bt, max_its=500):
    rank = AtA.shape[0]
    #AtA = A.T@A
    #AtY = A.T@Y
    
    tol = 1e-6**2
    initial_factor_diff = 0
    factor_diff = 1

    for i in range(max_its):
        if factor_diff < tol*initial_factor_diff and i > 2:
            break
        old_Bt = Bt.copy()
        factor_diff = 0

        for r in range(rank):
            delta_Bt = np.maximum((AtY[r] - AtA[r]@Bt)/AtA[r, r], -Bt[r])
            Bt[r] += delta_Bt

            factor_diff += delta_Bt@delta_Bt
            if np.all(Bt[r] == 0):
                Bt[r] = 1e-16*np.max(Bt)
        
        factor_diff = np.linalg.norm(Bt - old_Bt)
        if i == 0:
            initial_factor_diff = factor_diff
    return Bt


def prox_reg_nnls(AtA, AtY, Bt, reg_strength, proximity_center, max_its=500):
    rank = AtA.shape[0]
    
    #AtA = A.T@A
    #AtY = A.T@Y

    tol = 1e-6
    initial_factor_diff = 0
    factor_diff = 1

    for i in range(max_its):
        if factor_diff < tol*initial_factor_diff and i > 2:
            break
        old_Bt = Bt.copy()
        for r in range(rank):
            Bt_col = (AtY[r] - AtA[r]@Bt + AtA[r, r] * Bt[r] + reg_strength*proximity_center[r]) / (AtA[r, r] + reg_strength)
            Bt[r] = np.maximum(0, Bt_col)
            if np.all(Bt[r] == 0):
                Bt[r] = 1e-16*np.max(Bt)
        
        factor_diff = np.linalg.norm(Bt - old_Bt)
        if i == 0:
            initial_factor_diff = factor_diff
    return Bt
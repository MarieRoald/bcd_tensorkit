import sys

import h5py
import json
from pathlib import Path
from shutil import copy

import tenkit
import numpy as np
from block_tenkit_admm import double_splitting_parafac2, block_parafac2
from tqdm import trange


NOISE_PATH = "noise"
TENSOR_PATH = "evolving_tensor"
H5_GROUPS = ["dataset"]
SLICES_PATH = "dataset/tensor"

NOISE_LEVEL = 0.33
NUM_DATASETS = 50

INNER_TOL = 1e-3
INNER_SUB_ITS = 5

RELATIVE_TOLERANCE = 1e-10
ABSOLUTE_TOLERANCE = 1e-10
MAX_ITERATIONS = 1000

MIN_NODES = 3
MAX_NODES = 20

I = 20
J = 200
K = 40
RANK = 3

RIDGE_PENALTY = float(sys.argv[1])

OUTPUT_PATH = Path(
    f"201127_noise_{NOISE_LEVEL}_20_200_40_RIDGE_{RIDGE_PENALTY}".replace(".", "-")
)
DECOMPOSITION_FOLDER = OUTPUT_PATH/"decompositions"
DECOMPOSITION_FOLDER.mkdir(exist_ok=True, parents=True)
RESULTS_FOLDER = OUTPUT_PATH/"results"
RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)
CHECKPOINTS_FOLDER = OUTPUT_PATH/"checkpoints"
CHECKPOINTS_FOLDER.mkdir(exist_ok=True, parents=True)

def truncated_normal(size, rng):
    factor = rng.standard_normal(size)
    factor[factor < 0] = 0
    return factor

def generate_component(I, J, K, rank, rng):
    A = truncated_normal((I, rank), rng)
    blueprint_B = truncated_normal((J, rank), rng)
    B = [np.roll(blueprint_B, i) for i in range(K)]
    C = rng.uniform(0.1, 1.1, size=(K, rank))

    return tenkit.decomposition.EvolvingTensor(A, B, C)

def generate_noise(I, J, K, rng):
    return rng.standard_normal(size=(I, J, K))

def get_dataset_filename(dataset_num):
    return DECOMPOSITION_FOLDER/f"{dataset_num:03d}.h5"

def store_data(dataset_num, decomposition, noise):
    filename = get_dataset_filename(dataset_num)
    with h5py.File(filename, "w") as h5:
        for group_name in H5_GROUPS:
            h5.create_group(group_name)
        group = h5.create_group(TENSOR_PATH)
        decomposition.store_in_hdf5_group(group)
        h5[NOISE_PATH] = noise.transpose(2, 0, 1)
        h5[SLICES_PATH] = np.asarray(decomposition.construct_slices())

def add_noise(X, noise, noise_level):
    return X + noise_level*noise*np.linalg.norm(X)/np.linalg.norm(noise)

def create_loggers(dataset_num):
    dataset_filename = get_dataset_filename(dataset_num)
    loss_logger = tenkit.decomposition.logging.LossLogger()
    explained_variance_logger = tenkit.decomposition.logging.ExplainedVarianceLogger()
    relative_sse_logger = tenkit.decomposition.logging.RelativeSSELogger()
    sse_logger = tenkit.decomposition.logging.SSELogger()
    fms_logger = tenkit.decomposition.logging.EvolvingTensorFMSLogger(dataset_filename, "evolving_tensor")
    fmsA_logger = tenkit.decomposition.logging.EvolvingTensorFMSALogger(dataset_filename, "evolving_tensor")
    fmsB_logger = tenkit.decomposition.logging.EvolvingTensorFMSBLogger(dataset_filename, "evolving_tensor")
    fmsC_logger = tenkit.decomposition.logging.EvolvingTensorFMSCLogger(dataset_filename, "evolving_tensor")
    time_logger = tenkit.decomposition.logging.Timer()
    coupling_error_logger = tenkit.decomposition.logging.CouplingErrorLogger()
    rhoA_logger = tenkit.decomposition.logging.Parafac2RhoALogger()
    rhoB_logger = tenkit.decomposition.logging.Parafac2RhoBLogger()
    rhoC_logger = tenkit.decomposition.logging.Parafac2RhoCLogger()
    num_subits_logger_A = tenkit.decomposition.logging.NumSubIterationsLogger(0)
    num_subits_logger_B = tenkit.decomposition.logging.NumSubIterationsLogger(1)
    num_subits_logger_C = tenkit.decomposition.logging.NumSubIterationsLogger(2)

    return [
        loss_logger,
        explained_variance_logger,
        relative_sse_logger,
        sse_logger,
        fms_logger,
        fmsA_logger,
        fmsB_logger,
        fmsC_logger,
        coupling_error_logger,
        time_logger,
        rhoA_logger,
        rhoB_logger,
        rhoC_logger,
        num_subits_logger_A,
        num_subits_logger_B,
        num_subits_logger_C,
    ]

def run_double_experiment(dataset_num, X, rank):
    loggers = create_loggers(dataset_num)
    pf2 = double_splitting_parafac2.BlockEvolvingTensor(
        rank,
        sub_problems=[
            double_splitting_parafac2.Mode0ADMM(ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_its=INNER_SUB_ITS, tol=INNER_TOL),
            double_splitting_parafac2.DoubleSplittingParafac2ADMM(ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_it=INNER_SUB_ITS, tol=INNER_TOL),
            double_splitting_parafac2.Mode2ADMM(ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_its=INNER_SUB_ITS, tol=INNER_TOL),
        ],
        convergence_tol=RELATIVE_TOLERANCE,
        absolute_tol=ABSOLUTE_TOLERANCE,
        loggers=loggers,
        max_its=MAX_ITERATIONS,
        checkpoint_path=CHECKPOINTS_FOLDER/f"double_split_{dataset_num:03d}.h5",
        checkpoint_frequency=2000,
    )
    pf2.fit(X)
    return pf2

def run_double_rho_sum_experiment(dataset_num, X, rank):
    loggers = create_loggers(dataset_num)
    pf2 = double_splitting_parafac2.BlockEvolvingTensor(
        rank,
        sub_problems=[
            double_splitting_parafac2.Mode0ADMM(ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_its=INNER_SUB_ITS, tol=INNER_TOL),
            double_splitting_parafac2.DoubleSplittingParafac2ADMM_SeparatePF2Rho(ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_it=INNER_SUB_ITS, tol=INNER_TOL, rho_reduction=np.sum),
            double_splitting_parafac2.Mode2ADMM(ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_its=INNER_SUB_ITS, tol=INNER_TOL),
        ],
        convergence_tol=RELATIVE_TOLERANCE,
        absolute_tol=ABSOLUTE_TOLERANCE,
        loggers=loggers,
        max_its=MAX_ITERATIONS,
        checkpoint_path=CHECKPOINTS_FOLDER/f"double_split_rho_sum_{dataset_num:03d}.h5",
        checkpoint_frequency=2000,
    )
    pf2.fit(X)
    return pf2
    
def run_double_rho_max_experiment(dataset_num, X, rank):
    loggers = create_loggers(dataset_num)
    pf2 = double_splitting_parafac2.BlockEvolvingTensor(
        rank,
        sub_problems=[
            double_splitting_parafac2.Mode0ADMM(ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_its=INNER_SUB_ITS, tol=INNER_TOL),
            double_splitting_parafac2.DoubleSplittingParafac2ADMM_SeparatePF2Rho(ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_it=INNER_SUB_ITS, tol=INNER_TOL, rho_reduction=np.max),
            double_splitting_parafac2.Mode2ADMM(ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_its=INNER_SUB_ITS, tol=INNER_TOL),
        ],
        convergence_tol=RELATIVE_TOLERANCE,
        absolute_tol=ABSOLUTE_TOLERANCE,
        loggers=loggers,
        max_its=MAX_ITERATIONS,
        checkpoint_path=CHECKPOINTS_FOLDER/f"double_split_rho_max_{dataset_num:03d}.h5",
        checkpoint_frequency=2000,
    )
    pf2.fit(X)
    return pf2
    
    
def run_flexible_experiment(dataset_num, X, rank):
    loggers = create_loggers(dataset_num)
    pf2 = double_splitting_parafac2.BlockEvolvingTensor(
        rank,
        sub_problems=[
            double_splitting_parafac2.Mode0ADMM(non_negativity=True, max_its=INNER_SUB_ITS, tol=INNER_TOL),
            double_splitting_parafac2.FlexibleCouplingParafac2(),
            double_splitting_parafac2.Mode2ADMM(non_negativity=True, max_its=INNER_SUB_ITS, tol=INNER_TOL),
        ],
        convergence_tol=RELATIVE_TOLERANCE,
        absolute_tol=ABSOLUTE_TOLERANCE,
        loggers=loggers,
        max_its=MAX_ITERATIONS,
        checkpoint_path=CHECKPOINTS_FOLDER/f"flexible_coupling_{dataset_num:03d}.h5",
        checkpoint_frequency=2000,
        problem_order=(0, 1, 2),
        convergence_method="flex"
    )
    pf2.fit(X)
    return pf2

def run_single_C_experiment(dataset_num, X, rank):
    loggers = create_loggers(dataset_num)
    pf2 = block_parafac2.BlockParafac2(
        rank,
        sub_problems=[
            block_parafac2.ADMM(mode=0, ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_its=INNER_SUB_ITS, tol=INNER_TOL),
            block_parafac2.Parafac2ADMM(ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_it=INNER_SUB_ITS, tol=INNER_TOL),
            block_parafac2.ADMM(mode=2, ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_its=INNER_SUB_ITS, tol=INNER_TOL),
        ],
        convergence_tol=RELATIVE_TOLERANCE,
        absolute_tolerance=ABSOLUTE_TOLERANCE,
        loggers=loggers,
        projection_update_frequency=1,
        max_its=MAX_ITERATIONS,
        checkpoint_path=CHECKPOINTS_FOLDER/f"single_split_C{dataset_num:03d}.h5",
        checkpoint_frequency=2000,
    )
    pf2.fit(X)
    return pf2

def run_single_Dk_experiment(dataset_num, X, rank):
    loggers = create_loggers(dataset_num)
    pf2 = double_splitting_parafac2.BlockEvolvingTensor(
        rank,
        sub_problems=[
            double_splitting_parafac2.Mode0ADMM(ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_its=INNER_SUB_ITS, tol=INNER_TOL),
            double_splitting_parafac2.SingleSplittingParafac2ADMM(ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_it=INNER_SUB_ITS, tol=INNER_TOL),
            double_splitting_parafac2.Mode2ADMM(ridge_penalty=RIDGE_PENALTY, non_negativity=True, max_its=INNER_SUB_ITS, tol=INNER_TOL),
        ],
        convergence_tol=RELATIVE_TOLERANCE,
        absolute_tol=ABSOLUTE_TOLERANCE,
        loggers=loggers,
        max_its=MAX_ITERATIONS,
        checkpoint_path=CHECKPOINTS_FOLDER/f"single_split_Dk{dataset_num:03d}.h5",
        checkpoint_frequency=2000,
    )
    pf2.fit(X)
    return pf2

def generate_results(dataset_num, decomposer):
    loggers = decomposer.loggers
    results = {
        'iteration': loggers[0].log_iterations,
        'loss': loggers[0].log_metrics,
        'Fit': loggers[1].log_metrics,
        'Relative SSE': loggers[2].log_metrics,
        'SSE': loggers[3].log_metrics,
        'FMS': loggers[4].log_metrics,
        'FMS_A': loggers[5].log_metrics,
        'FMS_B': loggers[6].log_metrics,
        'FMS_C': loggers[7].log_metrics,
        'coupling_error': loggers[8].log_metrics,
        'time': loggers[9].log_metrics,
        'rho_A': loggers[10].log_metrics,
        'rho_B': loggers[11].log_metrics,
        'rho_C': loggers[12].log_metrics,
        'num_sub_iterations_A': loggers[13].log_metrics,
        'num_sub_iterations_B': loggers[14].log_metrics,
        'num_sub_iterations_C': loggers[15].log_metrics,
    }
    return results

def store_results(dataset_num, prefix, decomposer):
    results = generate_results(dataset_num, decomposer)
    with open(RESULTS_FOLDER/f"{prefix}_{dataset_num:03d}.json", "w") as f:
        json.dump(results, f)

def run_experiment(dataset_num):
    np.random.seed(0)
    rng = np.random.RandomState(dataset_num)
    decomposition = generate_component(I, J, K, RANK, rng)
    noise = generate_noise(I, J, K, rng)
    store_data(dataset_num, decomposition, noise)

    X = decomposition.construct_tensor()
    noisy_X = add_noise(X, noise, NOISE_LEVEL)

    double_pf2 = run_double_experiment(dataset_num, noisy_X, RANK)
    store_results(dataset_num, "double_split", double_pf2)

    double_sum_pf2 = run_double_rho_sum_experiment(dataset_num, noisy_X, RANK)
    store_results(dataset_num, "double_rho_sum_split", double_sum_pf2)

    double_max_pf2 = run_double_rho_max_experiment(dataset_num, noisy_X, RANK)
    store_results(dataset_num, "double_rho_max_split", double_max_pf2)

    #flexible_pf2 = run_flexible_experiment(dataset_num, noisy_X, RANK)
    #store_results(dataset_num, "flexible_coupling", flexible_pf2)

    #single_C_pf2 = run_single_C_experiment(dataset_num, noisy_X, RANK)
    #store_results(dataset_num, "single_split_C", single_C_pf2)

    single_Dk_pf2 = run_single_Dk_experiment(dataset_num, noisy_X, RANK)
    store_results(dataset_num, "single_split_Dk", single_Dk_pf2)

def run_experiment_on_existing_data(dataset_num):
    X = decomposition.construct_tensor()
    noisy_X = add_noise(X, noise, NOISE_LEVEL)

    double_pf2 = run_double_experiment(dataset_num, noisy_X, RANK)
    store_results(dataset_num, "double_split", double_pf2)

    #flexible_pf2 = run_flexible_experiment(dataset_num, noisy_X, RANK)
    #store_results(dataset_num, "flexible_coupling", flexible_pf2)

    #single_C_pf2 = run_single_C_experiment(dataset_num, noisy_X, RANK)
    #store_results(dataset_num, "single_split_C", single_C_pf2)

    single_Dk_pf2 = run_single_Dk_experiment(dataset_num, noisy_X, RANK)
    store_results(dataset_num, "single_split_Dk", single_Dk_pf2)


if __name__ == "__main__":
    np.random.seed(0)
    from joblib import delayed, Parallel
    Parallel(n_jobs=-1)(delayed(run_experiment)(i) for i in range(NUM_DATASETS))
    copy("timing_experiment.py", OUTPUT_PATH/"timing_experiment.py")    
    #for dataset_num in trange(NUM_DATASETS):
        #run_experiment(dataset_num)

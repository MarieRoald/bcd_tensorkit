import sys

import h5py
import json
from pathlib import Path
from shutil import copy

import tenkit
import numpy as np
from scipy import io
from block_tenkit_admm import double_splitting_parafac2, block_parafac2
from tqdm import trange


NOISE_PATH = "noise"
TENSOR_PATH = "evolving_tensor"
H5_GROUPS = ["dataset"]
SLICES_PATH = "dataset/tensor"

NOISE_LEVEL = float(sys.argv[1])
NUM_DATASETS = 50

INNER_TOL = 1e-3
INNER_SUB_ITS = 5

RELATIVE_TOLERANCE = 1e-10
ABSOLUTE_TOLERANCE = 1e-10
MAX_ITERATIONS = 1000

MIN_NODES = 3
MAX_NODES = 20

I = 20
J = 201
K = 40
RANK = 3

OUTPUT_PATH = Path(
    f"201204_noise_{NOISE_LEVEL}_20_200_40_NN".replace(".", "-")
)
DECOMPOSITION_FOLDER = OUTPUT_PATH/"decompositions"
DECOMPOSITION_FOLDER.mkdir(exist_ok=True, parents=True)
RESULTS_FOLDER = OUTPUT_PATH/"results"
RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)
CHECKPOINTS_FOLDER = OUTPUT_PATH/"checkpoints"
CHECKPOINTS_FOLDER.mkdir(exist_ok=True, parents=True)

RIDGE_PENALTY = float(sys.argv[2])
L2_SIMILARITY = np.zeros((J, J))
for j in range(J):
    if j > 0:
        L2_SIMILARITY[j, j] += 1
        L2_SIMILARITY[j, j-1] -= 1
    if j < J-1:
        L2_SIMILARITY[j, j] += 1
        L2_SIMILARITY[j, j+1] -= 1
L2_SIMILARITY *= float(sys.argv[3])

def truncated_normal(size, rng):
    factor = rng.standard_normal(size)
    factor[factor < 0] = 0
    return factor

def generate_component(I, J, K, rank, rng):
    A = truncated_normal((I, rank), rng)
    blueprint_B = io.loadmat("B.mat")["temp"]
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
    return {
        "Loss": tenkit.decomposition.logging.LossLogger(),
        "Fit": tenkit.decomposition.logging.ExplainedVarianceLogger(),
        "Relative SSE": tenkit.decomposition.logging.RelativeSSELogger(),
        "SSE": tenkit.decomposition.logging.SSELogger(),
        "FMS": tenkit.decomposition.logging.EvolvingTensorFMSLogger(dataset_filename, "evolving_tensor", fms_reduction="min"),
        "FMS A": tenkit.decomposition.logging.EvolvingTensorFMSALogger(dataset_filename, "evolving_tensor", fms_reduction="min"),
        "FMS B": tenkit.decomposition.logging.EvolvingTensorFMSBLogger(dataset_filename, "evolving_tensor", fms_reduction="min"),
        "FMS C": tenkit.decomposition.logging.EvolvingTensorFMSCLogger(dataset_filename, "evolving_tensor", fms_reduction="min"),
        "FMS (avg)": tenkit.decomposition.logging.EvolvingTensorFMSLogger(dataset_filename, "evolving_tensor", fms_reduction="mean"),
        "FMS A (avg)": tenkit.decomposition.logging.EvolvingTensorFMSALogger(dataset_filename, "evolving_tensor", fms_reduction="mean"),
        "FMS B (avg)": tenkit.decomposition.logging.EvolvingTensorFMSBLogger(dataset_filename, "evolving_tensor", fms_reduction="mean"),
        "FMS C (avg)": tenkit.decomposition.logging.EvolvingTensorFMSCLogger(dataset_filename, "evolving_tensor", fms_reduction="mean"),
        "Time": tenkit.decomposition.logging.Timer(),
        "Coupling error": tenkit.decomposition.logging.CouplingErrorLogger(),
        "Coupling error 0": tenkit.decomposition.logging.SingleCouplingErrorLogger(0),
        "Coupling error 1": tenkit.decomposition.logging.SingleCouplingErrorLogger(1),
        "Coupling error 2": tenkit.decomposition.logging.SingleCouplingErrorLogger(2),
        "Coupling error 3": tenkit.decomposition.logging.SingleCouplingErrorLogger(3),
        "Rho A": tenkit.decomposition.logging.Parafac2RhoALogger(),
        "Rho B": tenkit.decomposition.logging.Parafac2RhoBLogger(),
        "Rho C": tenkit.decomposition.logging.Parafac2RhoCLogger(),
        "Num subiterations A": tenkit.decomposition.logging.NumSubIterationsLogger(0),
        "Num subiterations B": tenkit.decomposition.logging.NumSubIterationsLogger(1),
        "Num subiterations C": tenkit.decomposition.logging.NumSubIterationsLogger(2),
        "Regulariser 0": tenkit.decomposition.logging.SingleModeRegularisationLogger(0),
        "Regulariser 1": tenkit.decomposition.logging.SingleModeRegularisationLogger(1),
        "Regulariser 2": tenkit.decomposition.logging.SingleModeRegularisationLogger(2),
    }

def run_smooth_experiment(dataset_num, X, rank):
    loggers = create_loggers(dataset_num)
    pf2 = double_splitting_parafac2.BlockEvolvingTensor(
        rank,
        sub_problems=[
            double_splitting_parafac2.Mode0ADMM(
                ridge_penalty=RIDGE_PENALTY,
                non_negativity=False,
                max_its=INNER_SUB_ITS,
                tol=INNER_TOL
            ),
            double_splitting_parafac2.DoubleSplittingParafac2ADMM(
                l2_similarity=L2_SIMILARITY,
                ridge_penalty=RIDGE_PENALTY,
                non_negativity=False,
                max_it=INNER_SUB_ITS,
                tol=INNER_TOL
            ),
            double_splitting_parafac2.Mode2ADMM(
                ridge_penalty=RIDGE_PENALTY,
                non_negativity=True,
                max_its=INNER_SUB_ITS,
                tol=INNER_TOL
            ),
        ],
        convergence_tol=RELATIVE_TOLERANCE,
        absolute_tol=ABSOLUTE_TOLERANCE,
        loggers=list(loggers.values()),
        max_its=MAX_ITERATIONS,
        checkpoint_path=CHECKPOINTS_FOLDER/f"double_split_{dataset_num:03d}.h5",
        checkpoint_frequency=2000,
    )
    pf2.fit(X)
    return pf2
    
def run_als_experiment(dataset_num, X, rank):
    loggers = create_loggers(dataset_num)
    pf2 = tenkit.decomposition.Parafac2_ALS(
        rank,
        non_negativity_constraints=[True, False, True],
        convergence_tol=RELATIVE_TOLERANCE,
        loggers=list(loggers.values()),
        max_its=MAX_ITERATIONS,
        checkpoint_path=CHECKPOINTS_FOLDER/f"als_{dataset_num:03d}.h5",
        checkpoint_frequency=2000,
        print_frequency=-1
    )
    pf2.fit(X)
    return pf2

def generate_results(dataset_num, decomposer):
    logger_names = create_loggers(dataset_num).keys()
    logs = [logger.log_metrics for logger in decomposer.loggers]
    results = dict(zip(logger_names, logs))
    results['iteration'] = decomposer.loggers[0].log_iterations
    return results
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

    pf2 = run_als_experiment(dataset_num, noisy_X, RANK)
    store_results(dataset_num, "als", pf2)

    flexible_pf2 = run_flexible_experiment(dataset_num, noisy_X, RANK)
    store_results(dataset_num, "flexible_coupling", flexible_pf2)


if __name__ == "__main__":
    np.random.seed(0)
    copy("timing_experiment.py", OUTPUT_PATH/"timing_experiment.py")    
    for dataset_num in trange(NUM_DATASETS):
        run_experiment(dataset_num)

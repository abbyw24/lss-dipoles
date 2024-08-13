import itertools
import numpy as np
from pathlib import Path

import generate_mocks as gm


def main():

    analyze_mocks()
    #analyze_data()

def analyze_mocks():

    dir_mocks = '../data/mocks'
    dir_results = '../results/results_mocks'
    Path.mkdir(Path(dir_results), exist_ok=True, parents=True)

    case_dicts = gm.case_set()
    n_trials_per_case = 12

    for case_dict in case_dicts:

        tag_case = f"_case{case_dict['Cell_mode']}-{case_dict['selfunc_mode']}-{case_dict['dipole_amp']}"

        for i in range(n_trials_per_case):

            fn_mock = f"{dir_mocks}/mock{tag_case}_trial{i}.npy"
            mock = np.load(fn_mock, allow_pickle=True)

            result = analyze(mock, case_dict)

            fn_res = f"{dir_results}/results_mock{tag_case}_trial{i}.npy"
            np.save(fn_res, result)


def analyze_data():

    dir_results = '../results/results_data'

    case_dict = {
        "selfunc_mode": 'fiducial',
    }
    qmap = # load data
    result = analyze(qmap, case_dict)

    fn_res = f"{dir_results}/results_quaia.npy"
    np.save(fn_res, result)


def analyze(qmap, case_dict):
    return "i'm an analysis!"



if __name__ == "__main__":
    main()
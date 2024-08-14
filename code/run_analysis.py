import itertools
import numpy as np
from pathlib import Path

import generate_mocks as gm
import dipole

def main():

    analyze_mocks()
    #analyze_data()

def analyze_mocks():

    dir_mocks = './data/mocks'
    dir_results = './data/results/results_mocks'
    Path.mkdir(Path(dir_results), exist_ok=True, parents=True)

    case_dicts = gm.case_set()
    n_trials_per_case = 2

    for case_dict in case_dicts:
        fns_res = []
        for i in range(n_trials_per_case):

            fn_mock = f"{dir_mocks}/mock{case_dict['tag']}_trial{i}.npy"
            print(f"analyze_mocks(): reading file {fn_mock}")
            mock = np.load(fn_mock, allow_pickle=True)

            Lambdas, comps = analyze(mock, case_dict)
            result_dict = {
                "Lambdas" : Lambdas,
                "dipole_comps" : comps
            }
            fn_res = os.path.join(dir_results, f"lambda_comps_mock{case_dict['tag']}_trial{i}.npy")
            print(f"analyze_mocks(): writing file {fn_res}")
            np.save(fn_res, result_dict)
            fns_res.append(fn_res)
            
        # make a plot of the trials in this case
        

# def analyze_data():
#     dir_results = '../results/results_data'
#     case_dict = {
#         "selfunc_mode": 'fiducial'
#     result = analyze(qmap, case_dict)
#     fn_res = f"{dir_results}/results_quaia.npy"
#     np.save(fn_res, result)

def analyze(qmap, case_dict):
    Lambdas = np.geomspace(1e-3, 1e0, 33)
    comps = np.zeros((len(Lambdas), 3))
    for i, Lambda in enumerate(Lambdas):
        comps[i] = dipole.measure_dipole_in_overdensity_map_Lambda(qmap,
                                                                   selfunc=gm.get_selfunc_map(case_dict['selfunc_mode']), Lambda=Lambda)
    return Lambdas, comps

if __name__ == "__main__":
    main()
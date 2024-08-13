import itertools
import numpy as np
from pathlib import Path


def main():
    generate_mocks_from_cases()


def case_set():

    Cell_modes = ['zeros', 'flat', 'datalike']
    selfunc_modes = ['ones', 'binary', 'datasf']
    dipole_amps = [0.0, 0.007, 0.014]

    arrs = [Cell_modes, selfunc_modes, dipole_amps]
    cases = list(itertools.product(*arrs))

    case_dicts = []
    for case in cases:
        case_dict = {
            "Cell_mode": case[0],
            "selfunc_mode": case[1],
            "dipole_amp": case[2]
        }
        case_dicts.append(case_dict)

    return case_dicts


def get_payload(case_dict):
    payload_dict = {
        "Cells": get_cells(case_dict['Cell_mode']), # write this function!
        "selfunc": get_selfunc(case_dict['selfunc_mode']), # write this function!
        "dipole_amp": case_dict['dipole_amp']
    }
    return payload_dict


def generate_mocks_from_cases():

    dir_mocks = '../data/mocks'
    Path.mkdir(Path(dir_mocks), exist_ok=True, parents=True)

    case_dicts = case_set()
    n_trials_per_case = 12

    for case_dict in case_dicts:
        
        tag_case = f"_case{case_dict['Cell_mode']}-{case_dict['selfunc']}-{case_dict['dipole_amp']}"
        payload = get_payload(case_dict) 

        for i in range(n_trials_per_case):

            mock = generate_mock(payload, trial=i) # or do you just want the case here and get payload inside genmock?

            fn_mock = f"{dir_mocks}/mock{tag_case}_trial{i}.npy"
            np.save(fn_mock, mock)


def generate_mock():
    return "i'm a mock!"


if __name__ == "__main__":
    main()
"""
# make plots of mocks and real data

## BUGS:
- file names must be synchronized here to `run_analysis.py`
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

import generate_mocks as gm


### MATPLOTLIB SETTINGS
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 16 
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['lines.linewidth'] = 2


def main():

    dir_figs = '../figures/figures_2024-09-23'
    Path.mkdir(Path(dir_figs), exist_ok=True, parents=True)

    Lambda1, Lambda2 = 1e-2, 1e-1

    set_name = 'full'
    #set_name = 'flat_quaia'
    tag_fig = f'_{set_name}'
    catalog_name = 'quaia_G20.0'
    selfunc_mode = 'quaia_G20.0_orig'
    case_dict_data = {
        "catalog_name": catalog_name,
        "selfunc_mode": selfunc_mode, #this also multiplies in the mask
        "tag": f"_case-{selfunc_mode}"
    }
    case_dicts_mock = gm.case_set(set_name=set_name)
    for case_dict_mock in case_dicts_mock:
        tag_fig = f"_{catalog_name}{case_dict_mock['tag']}"
        fn_fig = f'{dir_figs}/dipole_comps_vs_lambdas{tag_fig}.png'
        plot_dipole_comps_vs_lambdas(case_dict_data, case_dict_mock, 
                                    title=tag_fig[1:], fn_fig=fn_fig)

        fn_fig = f'{dir_figs}/Cells_Lambdas-{Lambda1:.1e}-{Lambda2:.1e}{tag_fig}.png'
        plot_Cells(case_dict_data, case_dict_mock, Lambda1, Lambda2,
                    title=tag_fig[1:], fn_fig=fn_fig)


def plot_dipole_comps_vs_lambdas(#fn_comps_data, fns_comps_mocks, 
                                 case_dict_data, case_dict_mock,
                                 label_data='Data', label_mock='Mock',
                                 title='', fn_fig=None):

    dir_results_data = '../results/results_data'
    dir_results_mocks = '../results/results_mocks'

    # Load data
    fn_comps_data = os.path.join(dir_results_data, f"dipole_comps_lambdas_{case_dict_data['catalog_name']}{case_dict_data['tag']}.npy")
    result_dict = np.load(fn_comps_data, allow_pickle=True).item()
    dipole_comps_data = result_dict['dipole_comps']
    lambdas_data = result_dict['Lambdas']

    # Load mock
    lambdas_mocks = []
    dipole_amps_mocks = []
    pattern = f"{dir_results_mocks}/dipole_comps*{case_dict_mock['tag']}*.npy"
    print(f"looking for {pattern}")
    fn_comps_mock = glob.glob(pattern)
    n_trials = len(fn_comps_mock)

    # not necessary in order, careful!
    for i in range(n_trials):
        result_dict = np.load(fn_comps_mock[i],  allow_pickle=True).item()
        dipole_amps_mock = np.linalg.norm(result_dict['dipole_comps'], axis=-1)
        dipole_amps_mocks.append(dipole_amps_mock)
        lambdas_mocks.append(result_dict['Lambdas'])
    dipole_amps_mocks = np.array(dipole_amps_mocks)

    # Compute the norm of the dipole components for the actual data
    dipole_amps_data = np.linalg.norm(dipole_comps_data, axis=1)

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas_data, dipole_amps_data, lw=3, color='k', label=label_data)

    # Plot each mock trial with light red lines
    for i in range(n_trials):        
        label = None
        if i == 0:
            label = label_mock
        plt.plot(lambdas_mocks[i], dipole_amps_mocks[i], color='lightcoral', linewidth=1,
                 label=label)

    # Plot the mean of the mock data with a dark red line
    dipole_amps_mock_mean = np.mean(dipole_amps_mocks, axis=0)
    plt.plot(lambdas_mocks[0], dipole_amps_mock_mean, color='red', linewidth=3, 
             label=label_mock+' mean')

    # Adding grid, labels and legend
    plt.grid(alpha=0.5, lw=0.5)
    plt.xscale('log')

    plt.xlabel(r'$\Lambda$')
    plt.ylabel(r'$\mathcal{D}$, dipole amplitude')
    plt.title(title)
    plt.legend()

    if fn_fig is not None:
        plt.savefig(fn_fig, bbox_inches='tight')
        print(f"Saved figure to {fn_fig}")


def plot_Cells(case_dict_data, case_dict_mock, Lambda1, Lambda2,
                label_data='Data', label_mock='Mock',
                title='', fn_fig=None, ms=7):
    
    dir_results_data = '../results/results_data'
    dir_results_mocks = '../results/results_mocks'

    # Create figure
    plt.figure(figsize=(10, 6))

    # plot params
    markers = ['v', '^']

    for j, Lambda in enumerate([Lambda1, Lambda2]):

        # Load data
        fn_comps_data = os.path.join(dir_results_data,
                            f"Cells_Lambda-{Lambda:.1e}_{case_dict_data['catalog_name']}{case_dict_data['tag']}.npy")
        result_dict = np.load(fn_comps_data, allow_pickle=True).item()
        Cells_data = result_dict['Cells']
        ells_data = result_dict['ells']

        # Load mock
        ells_mocks = []
        Cells_mocks = []
        pattern = f"{dir_results_mocks}/Cells_Lambda-{Lambda:.1e}*{case_dict_mock['tag']}*.npy"
        print(f"looking for {pattern}")
        fn_comps_mock = glob.glob(pattern)
        n_trials = len(fn_comps_mock)

        # not necessary in order, careful!
        for i in range(n_trials):
            result_dict = np.load(fn_comps_mock[i],  allow_pickle=True).item()
            Cells_mocks.append(result_dict['Cells'])
            ells_mocks.append(result_dict['ells'])
        Cells_mocks = np.array(Cells_mocks)

        # Plot data
        plt.plot(ells_data, Cells_data, color='k', ls='None', marker=markers[j], ms=ms,
                    label=label_data + r'$\Lambda=$'f'{Lambda:.1e}', zorder=100)

        # Plot each mock trial with light red lines
        for i in range(n_trials):        
            label = None
            if i == 0:
                label = label_mock
            plt.plot(ells_mocks[i], Cells_mocks[i], color='lightcoral', marker=markers[j], ms=ms,
                    ls='None', label=label)

        # Plot the mean of the mock data with a dark red line
        Cells_mock_mean = np.mean(Cells_mocks, axis=0)
        plt.plot(ells_mocks[0], Cells_mock_mean, color='red', marker=markers[j], ms=ms,
                ls='None', label=label_mock+' mean, 'r'$\Lambda=$'f'{Lambda:.1e}', zorder=50)

    # Adding grid, labels and legend
    plt.grid(alpha=0.5, lw=0.5)
    plt.yscale('log')

    plt.xlabel(r'Multipole $\ell$')
    plt.ylabel(r'$C(\ell)$')
    plt.title(title)
    plt.legend()

    if fn_fig is not None:
        plt.savefig(fn_fig, bbox_inches='tight')
        print(f"Saved figure to {fn_fig}")


if __name__ == '__main__':
    main()
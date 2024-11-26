"""
# make plots of mocks and real data

## BUGS:
- file names must be synchronized here to `run_analysis.py`
"""

import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
import os
from pathlib import Path
import datetime

import generate_mocks as gm
import tools
from dipole import cmb_dipole, get_dipole

RESULTDIR = '/scratch/aew492/lss-dipoles_results'


### MATPLOTLIB SETTINGS
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 16 
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['legend.fontsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['lines.linewidth'] = 2


def main():

    dir_figs = os.path.join(RESULTDIR, 'figures/figures_' + str(datetime.date.today()))
    Path.mkdir(Path(dir_figs), exist_ok=True, parents=True)

    Lambda1, Lambda2 = 1e-2, 1e-1 # magic: corresponds to the Lambdas in run_analysis.py !

    set_name = 'ideal_catwise'
    tag_fig = f'_{set_name}'
    catalog_name = 'catwise' #'quaia_G20.0'
    selfunc_mode = 'catwise_zodi' #'quaia_G20.0_orig'
    case_dict_data = {
        "catalog_name": catalog_name,
        "selfunc_mode": selfunc_mode, #this also multiplies in the mask
        "tag": f"_case-{selfunc_mode}"
    }
    case_dicts_mock = gm.case_set(set_name=set_name)
    for case_dict_mock in case_dicts_mock:
        tag_fig = f"_{catalog_name}{case_dict_mock['tag']}"
        plot_recovered_dipoles(case_dict_data, case_dict_mock, Lambda1,
                                fn_fig=f'{dir_figs}/dipole_directions{tag_fig}.png')
        fn_fig = f'{dir_figs}/dipole_comps_vs_Lambdas{tag_fig}.png'
        plot_dipole_comps_vs_Lambdas(case_dict_data, case_dict_mock, 
                                    title=tag_fig[1:], fn_fig=fn_fig)

        # fn_fig = f'{dir_figs}/Cells_Lambdas-{Lambda1:.1e}-{Lambda2:.1e}{tag_fig}.png'
        # plot_Cells(case_dict_data, case_dict_mock, Lambda1, Lambda2,
        #             title=tag_fig[1:], fn_fig=fn_fig)


def plot_dipole_comps_vs_Lambdas(#fn_comps_data, fns_comps_mocks, 
                                 case_dict_data, case_dict_mock,
                                 title='', fn_fig=None):

    dir_results_data = os.path.join(RESULTDIR, 'results/results_data')
    dir_results_mocks = os.path.join(RESULTDIR, 'results/results_mocks')

    # Load data
    fn_comps_data = os.path.join(dir_results_data, f"dipole_comps_Lambdas_{case_dict_data['catalog_name']}{case_dict_data['tag']}.npy")
    result_dict = np.load(fn_comps_data, allow_pickle=True).item()
    dipole_comps_data = result_dict['dipole_comps']
    Lambdas_data = result_dict['Lambdas']

    # Load mock
    Lambdas_mocks = []
    dipole_amps_mocks = []
    pattern = f"{dir_results_mocks}/dipole_comps_lambdas*{case_dict_mock['tag']}*.npy"
    print(f"looking for {pattern}...")
    fn_comps_mock = glob.glob(pattern)
    n_trials = len(fn_comps_mock)
    print(f"found {n_trials} files with this pattern")

    # not necessary in order, careful!
    for i in range(n_trials):
        result_dict = np.load(fn_comps_mock[i],  allow_pickle=True).item()
        dipole_amps_mock = np.linalg.norm(result_dict['dipole_comps'], axis=-1)
        dipole_amps_mocks.append(dipole_amps_mock)
        Lambdas_mocks.append(result_dict['Lambdas'])
    dipole_amps_mocks = np.array(dipole_amps_mocks)

    # Compute the norm of the dipole components for the actual data
    dipole_amps_data = np.linalg.norm(dipole_comps_data, axis=1)

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.plot(Lambdas_data, dipole_amps_data, lw=3, color='k', label='Data', zorder=100)

    # Plot each mock trial with light red lines
    for i in range(n_trials):        
        label = 'Mock' if i==0 else ''
        plt.plot(Lambdas_mocks[i], dipole_amps_mocks[i], color='lightcoral', linewidth=0.5,
                 label=label)

    # Plot the input dipole amplitude for the mocks
    plt.axhline(case_dict_mock['dipole_amp'], color='red', ls='--', alpha=0.8, label='Input dipole amp.')

    # Plot the mean of the mock data with a dark red line
    dipole_amps_mock_mean = np.mean(dipole_amps_mocks, axis=0)
    plt.plot(Lambdas_mocks[0], dipole_amps_mock_mean, color='red', linewidth=3, 
             label=f'Mean of {n_trials} mocks', zorder=10)
    # Also plot the 1sigma
    dipole_amps_mock_std = np.std(dipole_amps_mocks, axis=0)
    # plt.plot(Lambdas_mocks[0], dipole_amps_mock_mean - dipole_amps_mock_std,
    #                     color='red', alpha=0.5)
    # plt.plot(Lambdas_mocks[0], dipole_amps_mock_mean + dipole_amps_mock_std,
    #                     color='red', alpha=0.5)
    plt.fill_between(Lambdas_mocks[0], dipole_amps_mock_mean - dipole_amps_mock_std,
                        dipole_amps_mock_mean + dipole_amps_mock_std,
                        color='darkorange', alpha=0.4, label=r'1$\sigma$')

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
                title='', fn_fig=None, ms=7, max_mocks=12):
    
    dir_results_data = os.path.join(RESULTDIR, 'results/results_data')
    dir_results_mocks = os.path.join(RESULTDIR, 'results/results_mocks')

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
        fn_comps_mock = tools.filter_max_mocks(fn_comps_mock, max_mocks)
        n_trials = len(fn_comps_mock)

        # not necessary in order, careful!
        for i in range(n_trials):
            result_dict = np.load(fn_comps_mock[i],  allow_pickle=True).item()
            Cells_mocks.append(result_dict['Cells'])
            ells_mocks.append(result_dict['ells'])
        Cells_mocks = np.array(Cells_mocks)

        # Plot data
        plt.plot(ells_data, Cells_data, mec='k', c='None', ls='None', marker=markers[j], ms=ms,
                    label=r'Data, $\Lambda=$'f'{Lambda:.1e}', zorder=100)

        # Plot each mock trial with light red lines
        for i in range(n_trials):        
            label = r'Mock, $\Lambda=$'f'{Lambda:.1e}' if i==0 else ''
            plt.plot(ells_mocks[i], Cells_mocks[i], color='lightcoral', marker=markers[j], ms=ms,
                    ls='None', label=label)

        # Plot the mean of the mock data with a dark red line
        Cells_mock_mean = np.mean(Cells_mocks, axis=0)
        plt.plot(ells_mocks[0], Cells_mock_mean, color='red', marker=markers[j], ms=ms,
                ls='None', label='Mock mean, 'r'$\Lambda=$'f'{Lambda:.1e}', zorder=50)

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


def plot_recovered_dipoles(case_dict_data, case_dict_mock, Lambda, fn_fig=None):

    dir_results_data = os.path.join(RESULTDIR, 'results/results_data')
    dir_results_mocks = os.path.join(RESULTDIR, 'results/results_mocks')

    # Load data
    fn_comps_data = os.path.join(dir_results_data, f"dipole_comps_Lambdas_{case_dict_data['catalog_name']}{case_dict_data['tag']}.npy")
    result_dict = np.load(fn_comps_data, allow_pickle=True).item()
    dipole_comps_data = result_dict['dipole_comps']
    Lambdas_data = result_dict['Lambdas']

    # index of the Lambda closest to input
    iLambda = np.argmin(np.abs(Lambdas_data[0]-Lambda))
    # recovered dipole amplitude and direction in the data
    dipole_amp_data, dipole_dir_data = get_dipole(dipole_comps_data[iLambda,:])

    # Load mocks
    Lambdas_mocks = []
    dipole_amps_mocks = []
    pattern = f"{dir_results_mocks}/dipole_comps_lambdas*{case_dict_mock['tag']}*.npy"
    print(f"looking for {pattern}...")
    fn_comps_mock = glob.glob(pattern)
    n_trials = len(fn_comps_mock)
    print(f"found {n_trials} files with this pattern")

    # gather the recovered components
    dipole_comps_mocks = []
    Lambdas_mocks = []
    # not necessary in order, careful!
    for i in range(n_trials):
        result_dict = np.load(fn_comps_mock[i],  allow_pickle=True).item()
        dipole_comps_mock = result_dict['dipole_comps'] # shape (len(Lambdas), 3)
        dipole_comps_mocks.append(dipole_comps_mock)
        Lambdas_mocks.append(result_dict['Lambdas'])
    dipole_comps_mocks = np.array(dipole_comps_mocks)
    Lambdas_mocks = np.array(Lambdas_mocks)

    # get the amplitudes and directions of the recovered dipole components
    dipole_amps_mocks = []
    dipole_dirs_mocks = []
    for i in range(n_trials):
        dipole_amps_ = []
        dipole_dirs_ = []
        for j in range(len(Lambdas_mocks[0])):
            dipole_amp, dipole_dir = get_dipole(dipole_comps_mocks[i,j])
            dipole_amps_.append(dipole_amp)
            dipole_dirs_.append(dipole_dir)
        dipole_amps_mocks.append(dipole_amps_)
        dipole_dirs_mocks.append(dipole_dirs_)
    dipole_amps_mocks = np.array(dipole_amps_mocks) # shape (ntrials, len(Lambdas))

    # (theta, phi) recovered for each mock using the input Lambda
    dipole_dirs_mocks_spherical = np.empty((n_trials, 2))
    for i, dipdirs in enumerate(dipole_dirs_mocks):
        dipdir = dipdirs[iLambda]
        theta, phi = tools.lonlat_to_thetaphi(dipdir.galactic.l, dipdir.galactic.b)
        dipole_dirs_mocks_spherical[i] = [theta.value, phi.value]
    
    # mean and standard deviation in (theta, phi)
    mean_theta_mocks, mean_phi_mocks = np.mean(dipole_dirs_mocks_spherical, axis=0)
    std_theta_mocks, std_phi_mocks = np.std(dipole_dirs_mocks_spherical, axis=0)

    # # histogram of the recovered dipole amplitudes
    # fig, ax = plt.subplots(figsize=(7,5))
    # _, _, _ = ax.hist(dipole_amps_mocks[:,iLambda], bins=20, color='grey', alpha=0.6)
    # ax.grid(alpha=0.5, lw=0.5)
    # ax.axvline(case_dict_mock['dipole_amp'], c='k', alpha=0.5, label='Input amplitude')
    # ax.set_xlabel('Dipole amplitude')
    # ax.set_ylabel('Mocks')
    # ax.legend()
    # ax.set_title('Recovered dipole amplitudes ('r'$\Lambda=$'f'{Lambdas_mocks[0][iLambda]:.5f}): '+case_dict_mock['tag'][1:])

    # expected/input direction
    input_dir = cmb_dipole()[1]

    # sky plot of the recovered dipole directions: color by recovered amplitude
    norm = mpl.colors.Normalize(vmin=min(dipole_amps_mocks[:,iLambda]), vmax=max(dipole_amps_mocks[:,iLambda]))
    smap = mpl.cm.ScalarMappable(norm=norm, cmap='cool')
    fig = plt.figure(figsize=(14,5))
    ax = fig.add_subplot(111, projection='mollweide')
    ax.grid(alpha=0.5, lw=0.5)

    # input dipole
    theta, phi = tools.lonlat_to_thetaphi(input_dir.galactic.l, input_dir.galactic.b)
    ax.scatter(phi.wrap_at(np.pi * u.rad), np.pi/2 * u.rad - theta, alpha=0.8, c='k', s=14, zorder=999)

    # data dipole
    theta, phi = tools.lonlat_to_thetaphi(dipole_dir_data.galactic.l, dipole_dir_data.galactic.b)
    ax.scatter(phi.wrap_at(np.pi * u.rad), np.pi/2 * u.rad - theta, alpha=0.8, c='r', s=14, zorder=1000)

    # recovered directions *galactic coords*
    for i, dipdirs in enumerate(dipole_dirs_mocks):
        dipdir = dipdirs[iLambda]
        theta, phi = tools.lonlat_to_thetaphi(dipdir.galactic.l, dipdir.galactic.b)
        ax.scatter(phi.wrap_at(np.pi * u.rad), np.pi/2 * u.rad - theta, alpha=0.4, s=14,
                    c=smap.to_rgba(dipole_amps_mocks[i, iLambda]))
    # mean recovered direction in mocks
    ax.scatter(Angle(mean_phi_mocks * u.rad).wrap_at(np.pi * u.rad), np.pi/2 * u.rad - Angle(mean_theta_mocks * u.rad), alpha=0.4, s=10,
                    marker='x', c='grey')
    # 1sigma contour for mocks
    ellipse = mpl.patches.Ellipse((Angle(mean_phi_mocks * u.rad).to(u.rad).wrap_at(np.pi*u.rad).value,
                            Angle(mean_theta_mocks * u.rad).to(u.rad).wrap_at(np.pi*u.rad).value),
                            width=Angle(2 * std_phi_mocks * u.rad).to(u.rad).wrap_at(np.pi*u.rad).value,
                            height=Angle(2 * std_theta_mocks * u.rad).to(u.rad).wrap_at(np.pi*u.rad).value,
                            color='grey', alpha=0.3)
    ax.add_artist(ellipse)
    fig.suptitle('Recovered dipole directions ('r'$\Lambda=$'f'{Lambdas_mocks[0][iLambda]:.5f}): '+case_dict_mock['tag'][1:])

    fig.colorbar(smap, ax=ax, label='Dipole amplitude')

    if fn_fig is not None:
        plt.savefig(fn_fig, bbox_inches='tight')
        print(f"Saved figure to {fn_fig}")


if __name__ == '__main__':
    main()

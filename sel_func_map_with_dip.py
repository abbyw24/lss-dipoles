"""
GOAL: Model the dipole in the selection function.

SF = GP(=Kate's maps) + mean function(=monopole+dipole)

"""


import numpy as np
import time
import os
import sys

from scipy.optimize import minimize
import healpy as hp
import george
from george.modeling import Model

from dipole import dipole

sys.path.insert(0, '/home/aew492/gaia-quasars-lss/code')
import utils
import masks
import maps


def main():
    print("starting selection function", flush=True)

    # parameters
    map_names = ['dust', 'stars', 'm10', 'mcs']
    NSIDE = 64
    G_max = 20.5
    fit_with_mask_mcs = False
    x_scale_name = 'zeromean'
    y_scale_name = 'log'
    fit_zeros = False
    shorten_arrays = False
    nshort = 1000

    # save file
    data_dir = '/scratch/aew492/quasars'
    maps_dir = os.path.join(data_dir, 'maps')
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)
    fn_prob = os.path.join(maps_dir, f'/scratch/aew492/quasars/maps/selection_function_NSIDE{NSIDE}_G{G_max}_dipole.fits')
    overwrite = True

    start = time.time()

    ## LOAD DATA
    print("Loading data", flush=True)
    cat_dir = os.path.join(data_dir, 'catalogs')
    if not os.path.exists(cat_dir):
        os.makedirs(cat_dir)
    fn_gaia = os.path.join(cat_dir, f'catalog_G{G_max}.fits')
    tab_gaia = utils.load_table(fn_gaia)

    # make healpy maps -> shape==(NPIX,), N determined by NSIDE
    print("Making QSO map", flush=True)
    maps_forsel = load_maps(NSIDE, map_names)
    map_nqso_data, _ = maps.get_map(NSIDE, tab_gaia['ra'], tab_gaia['dec'], null_val=0)


    ## CONSTRUCT FULL ARRAYS
    print("Constructing X and y", flush=True)
    NPIX = hp.nside2npix(NSIDE)
    X_train_full = construct_X(NPIX, map_names, maps_forsel)
    y_train_full = map_nqso_data
    y_train_full = y_train_full.astype(float)  # need this because will be inserting small vals where zero
    y_err_train_full = np.sqrt(y_train_full)  # assume poisson error


    ## INDICES TO FIT
    print("Getting indices to fit", flush=True)
    print("full map:", len(y_train_full))
    # should i do this in fitter??
    if fit_zeros:
        if y_scale_name=='log':
            idx_zero = np.abs(y_train_full) < 1e-4
            print('num zeros:', np.sum(idx_zero))
            y_train_full[idx_zero] = 0.5       # set zeros to 1/2 a star
        idx_fit = np.full(len(y_train_full), True)
        print('min post', np.min(y_train_full), flush=True)
    else:
        idx_fit = y_train_full > 0
    print("idx_fit:", len(idx_fit))

    if fit_with_mask_mcs:
        mask_mcs = masks.magellanic_clouds_mask(NSIDE)
        idx_nomcs = ~mask_mcs #because mask is 1 where masked
        # i think nomcs is breaking everything! #TODO check
        idx_fit = idx_fit & idx_nomcs

    X_train = X_train_full[idx_fit]
    y_train = y_train_full[idx_fit]
    y_err_train = y_err_train_full[idx_fit]
    print(np.min(y_train))

    if shorten_arrays:
        print("Shortening arrays for fit")
        idx = np.arange(nshort)
        X_train = X_train[idx]
        y_train = y_train[idx]
        y_err_train = y_err_train[idx]
        print("idx:", len(idx))


    ## TRAIN FITTER
    print("Training fitter", flush=True)
    print("X_train:", X_train.shape, "y_train:", y_train.shape, flush=True)
    fitter = FitterGP(X_train, y_train, y_err_train, 
                      x_scale_name=x_scale_name, y_scale_name=y_scale_name)
    fitter.train()
    # predict: the expected QUaia data
    print("Predicting", flush=True)
    y_pred = fitter.predict(X_train)
    print("y_pred:", len(y_pred))

    y_pred_full = np.zeros(y_train_full.shape)
    y_pred_full[idx] = y_pred

    # get rms error between the training set ("truth" QUaia map) and the predicted set (expected QUaia map)
    print('RMSE:', utils.compute_rmse(y_pred_full, y_train_full), flush=True)

    # make probability map
    print("Making probability map", flush=True)
    map_prob = map_expected_to_probability(y_pred_full, y_train_full, map_names, maps_forsel)

    # fails without proper len(map_prob)
    hp.write_map(fn_prob, map_prob, overwrite=overwrite)
    print(f"Saved map to {fn_prob}!", flush=True)

    end = time.time()
    print(f"Time: {end-start} s ({(end-start)/60.} min)", flush=True)


#hack! better way?
def map_expected_to_probability(map_expected, map_true, map_names, maps_forsel):
    # clean pixels = no significant contamination from any map
    idx_clean = np.full(len(map_expected), True)
    # get the "clean" indices from each map
    for map_name, map in zip(map_names, maps_forsel):
        if map_name=='dust':
            idx_map = map < 0.03
        elif map_name=='stars':
            idx_map = map < 15
        elif map_name=='m10':
            idx_map = map > 21
        elif map_name=='mcs':
            idx_map = map < 1 #mcs map has 0s where no mcs, tho this should be redundant w stars
        idx_clean = idx_clean & idx_map
    print("Number of clean healpixels:", np.sum(idx_clean), f"(Total: {len(map_expected)})")
    # average quasar density in the clean pixels
    nqso_clean = np.mean(map_true[idx_clean])

    # normalize expected by average density -> probability map
    map_prob = map_expected / nqso_clean
    # since probability, max value is 1
    map_prob[map_prob>1.0] = 1.0
    assert np.all(map_prob <= 1.0) and np.all(map_prob >= 0.0), "Probabilities must be <=1 and >=0!"
    return map_prob

# def load_maps(NSIDE, map_names):
#     maps_forsel = []
#     # TODO: should be writing these maps with hp.write_map() to a fits file!
#     map_functions = {'stars': maps.get_star_map,
#                      'dust': maps.get_dust_map,
#                      'm10': maps.get_m10_map,
#                      'mcs': maps.get_mcs_map}

#     for map_name in map_names:
#         maps_dir = '/scratch/aew492/quasars/maps'
#         if not os.path.exists(maps_dir):
#             os.makedirs(maps_dir)
#         fn_map = os.path.join(maps_dir, f'fmap_{map_name}_NSIDE{NSIDE}.npy')
#         maps_forsel.append( map_functions[map_name](NSIDE=NSIDE, fn_map=fn_map) )
#     return maps_forsel

### ABBY'S VERSION
def load_maps(NSIDE, map_names, maps_dir='/scratch/aew492/quasars/maps'):
    maps_forsel = []
    for map_name in map_names:
        map_fn = os.path.join(maps_dir, f'map_{map_name}_NSIDE{NSIDE}.npy')
        assert os.path.exists(map_fn), f"map {map_name} not found!"
        maps_forsel.append(np.load(map_fn))
    return maps_forsel

def f_dust(map_d):
    return map_d

def f_stars(map_s):
    return np.log(map_s)

def f_m10(map_m):
    return map_m

def f_mcs(map_mcs):
    map_mcs = map_mcs.astype(float)
    i_zeroorneg = map_mcs < 1e-4
    map_mcs[i_zeroorneg] = 1e-4
    return np.log(map_mcs)

def construct_X(NPIX, map_names, maps_forsel):
    f_dict = {'dust': f_dust,
             'stars': f_stars,
             'm10': f_m10,
             'mcs': f_mcs}
    X = np.vstack([f_dict[map_name](map) for map_name, map in zip(map_names, maps_forsel)])
    return X.T


class Fitter():

    def __init__(self, X_train, y_train, y_err_train, x_scale_name=None, y_scale_name=None):
        # TODO: add asserts that these are the right shapes
        self.X_train = X_train
        self.y_train = y_train
        self.y_err_train = y_err_train
        # TODO: add asserts that these are implemented
        self.x_scale_name = x_scale_name
        self.y_scale_name = y_scale_name

        self.X_train_scaled = self.scale_X(self.X_train)
        self.y_train_scaled = self.scale_y(self.y_train)
        self.y_err_train_scaled = self.scale_y_err(self.y_err_train)

    def scale_y_err(self, y_err):
        if self.y_scale_name=='log':
            return y_err / self.y_train
        else:
            # if not log, make sure no zeros; set min to 1
            #hack!
            y_err = np.clip(y_err, 1, None)
        return y_err

    def scale_X(self, X):
        X_scaled = X.copy()
        if self.x_scale_name=='zeromean':
            X_scaled -= np.mean(X_scaled, axis=0)
        return X_scaled

    def scale_y(self, y):
        y_scaled = y.copy()
        if self.y_scale_name=='log':
            y_scaled = np.log(y)
        return y_scaled

    def unscale_y(self, y_scaled):
        y_unscaled = y_scaled.copy()
        if self.y_scale_name=='log':
            y_unscaled = np.exp(y_scaled)
        return y_unscaled

    def train(self):
        pass

    def predict(self, X_pred):
        pass


class FitterGP(Fitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def train(self):
        ndim = self.X_train.shape[1]
        n_params = self.X_train_scaled.shape[1]
        print("n params:", n_params)
        kernel_p0 = np.exp(np.full(n_params, 0.1))
        kernel = george.kernels.ExpSquaredKernel(kernel_p0, ndim=ndim)
        mean_p0 = [2., 0., 0., 0.] # monopole + 3 dipole amplitudes
        self.gp = george.GP(kernel, mean=DipoleModel(*mean_p0), fit_mean=True)
        print('p init:', self.gp.get_parameter_vector())
        # pre-compute the covariance matrix and factorize it for a set of times and uncertainties
        self.gp.compute(self.X_train_scaled, self.y_err_train_scaled)
        print('p compute:', self.gp.get_parameter_vector())
        print('lnlike compute:', self.gp.log_likelihood(self.y_train_scaled))

        def neg_ln_like(p):
            self.gp.set_parameter_vector(p)
            return -self.gp.log_likelihood(self.y_train_scaled)

        def grad_neg_ln_like(p):
            self.gp.set_parameter_vector(p)
            return -self.gp.grad_log_likelihood(self.y_train_scaled)

        print("Minimizing")
        result = minimize(neg_ln_like, self.gp.get_parameter_vector(), jac=grad_neg_ln_like)
        print(result)
        self.gp.set_parameter_vector(result.x)
        print('p post op:', self.gp.get_parameter_vector())
        print('lnlike final:', self.gp.log_likelihood(self.y_train_scaled))

    
    def predict(self, X_pred):
        X_pred_scaled = self.scale_X(X_pred)
        print('predict p:', self.gp.get_parameter_vector())
        y_pred_scaled, _ = self.gp.predict(self.y_train_scaled, X_pred_scaled)
        return self.unscale_y(y_pred_scaled)


class DipoleModel(Model):
    parameter_names = ('monopole', 'dipole_x', 'dipole_y', 'dipole_z')
    # figure out how to pass NSIDE!
    NSIDE = 64
    NPIX = hp.nside2npix(NSIDE)
    thetas, phis = hp.pix2ang(NSIDE, ipix=np.arange(NPIX))
    
    def get_value(self, X):
        idx = np.arange(X.shape[0])
        return self.monopole + dipole(DipoleModel.thetas[idx], DipoleModel.phis[idx],
                                      self.dipole_x, self.dipole_y, self.dipole_z)
    
    def set_vector(self, v):
        self.monopole, self.dipole_x, self.dipole_y, self.dipole_z = v


if __name__=='__main__':
    main()
import numpy as np
import time
from datetime import datetime
import os
import sys

from scipy.optimize import minimize
import healpy as hp
import george
from george.modeling import Model

sys.path.insert(0, '/home/aew492/gaia-quasars-lss/code')
import utils
import masks
import maps


def main():
    print("Starting selection function", flush=True)

    # parameters
    map_names = ['dust', 'stars', 'm10', 'mcs']
    NSIDE = 64
    G_max = 20.5
    x_scale_name = 'zeromean'
    y_scale_name = 'log'
    fit_zeros = False
    shorten_arrays = False
    nshort = 1000
    random_pix = True
    save_map = True
    save_res = True

    # save file
    data_dir = '/scratch/aew492/quasars'
    maps_dir = os.path.join(data_dir, 'maps')
    if not os.path.exists(maps_dir):
        os.makedirs(maps_dir)
    shorttag = f'_{nshort}pix' if shorten_arrays else ''
    maptag = f'_{map_names[0]}only' if len(map_names)==1 else ''
    fn_prob = os.path.join(maps_dir, f'/scratch/aew492/quasars/maps/selection_function_NSIDE{NSIDE}_G{G_max}_nodipole{shorttag}{maptag}')
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
    print("full map:", len(y_train_full), flush=True)
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
        print("removed zeros ->", np.sum(idx_fit), flush=True)
    
    if shorten_arrays:
        idx_short = np.full(len(y_train_full), False)
        if random_pix:
            idx_rand = np.random.choice(NPIX, size=nshort)
            idx_short[idx_rand] = True
        else:
            idx_short[:nshort] = True
        idx_fit = idx_fit & idx_short
        print("shortened arrays ->", np.sum(idx_fit), flush=True)

    X_train = X_train_full[idx_fit]
    y_train = y_train_full[idx_fit]
    y_err_train = y_err_train_full[idx_fit]
    print(np.min(y_train), flush=True)
    # assert np.min(y_train)==1.


    ## TRAIN FITTER
    print("Training fitter", flush=True)
    print("X_train:", X_train.shape, "y_train:", y_train.shape, flush=True)
    fitter = FitterGP(X_train, y_train, y_err_train, 
                      x_scale_name=x_scale_name, y_scale_name=y_scale_name)
    fitter.train(maxiter=None)
    # predict: the expected QUaia data
    print("Predicting", flush=True)
    y_pred = fitter.predict(X_train)
    print("y_pred:", len(y_pred), flush=True)

    y_pred_full = np.zeros(y_train_full.shape)
    y_pred_full[idx_fit] = y_pred

    # get rms error between the training set ("truth" QUaia map) and the predicted set (expected QUaia map)
    print('RMSE:', utils.compute_rmse(y_pred_full, y_train_full), flush=True)

    # make probability map
    print("Making probability map", flush=True)
    map_prob = map_expected_to_probability(y_pred_full, y_train_full, map_names, maps_forsel)

    if save_res:
        res_fn = fn_prob+'-res'
        np.save(res_fn, fitter.result)
        print(f"Saved optimization results to {res_fn}!", flush=True)

    if save_map:
        fn_prob += ".fits"
        hp.write_map(fn_prob, map_prob, overwrite=overwrite)
        print(f"Saved map to {fn_prob}!", flush=True)

    end = time.time()
    print(f"Time: {end-start:.2f} s ({(end-start)/60.:.2f} min)", flush=True)


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
        assert X_train.shape[0]==y_train.shape[0]==y_err_train.shape[0], "check input array sizes!"
        self.X_train = X_train
        self.y_train = y_train
        self.y_err_train = y_err_train
        # TODO: add asserts that these are implemented
        self.x_scale_name = x_scale_name
        self.y_scale_name = y_scale_name

        self.X_train_scaled = self.scale_X(self.X_train)
        self.y_train_scaled = self.scale_y(self.y_train)
        self.y_err_train_scaled = self.scale_y_err(self.y_err_train)

        # for callback
        self.num_calls = 0 # how many times the likelihood function has been called
        self.callback_count = 0 # number of times callback has been called (measures iteration count)
        self.inputs = [] # input of all calls
        self.lnlikes = [] # result of all calls

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


    def train(self, maxiter=None):
        ndim = self.X_train.shape[1]
        n_params = self.X_train_scaled.shape[1]
        print("n params:", n_params)
        kernel_p0 = np.exp(np.full(n_params, 0.1))
        kernel = george.kernels.ExpSquaredKernel(kernel_p0, ndim=ndim)
        self.gp = george.GP(kernel, mean=np.mean(self.y_train_scaled), fit_mean=True)
        print('p init:', self.gp.get_parameter_vector())
        # pre-compute the covariance matrix and factorize it for a set of times and uncertainties
        self.gp.compute(self.X_train_scaled, self.y_err_train_scaled)
        print('p compute:', self.gp.get_parameter_vector())
        print('lnlike compute:', self.gp.log_likelihood(self.y_train_scaled))

        def neg_ln_like(p):
            self.gp.set_parameter_vector(p)
            lnlike = self.gp.log_likelihood(self.y_train_scaled)
            self.inputs.append(p)
            self.lnlikes.append(lnlike)
            self.num_calls += 1
            return -lnlike

        def grad_neg_ln_like(p):
            self.gp.set_parameter_vector(p)
            return -self.gp.grad_log_likelihood(self.y_train_scaled)

        print("Minimizing", flush=True)
        print("current time:", datetime.now().strftime("%H:%M:%S"), flush=True)

        result = minimize(neg_ln_like, self.gp.get_parameter_vector(), jac=grad_neg_ln_like, callback=self.callback,
                            options=dict(maxiter=maxiter, disp=True))
        self.result = result
        self.gp.set_parameter_vector(result.x)
        print('p post op:', self.gp.get_parameter_vector())
        # print('lnlike final:', self.gp.log_likelihood(self.y_train_scaled))
    
    def predict(self, X_pred):
        X_pred_scaled = self.scale_X(X_pred)
        # print('predict p:', self.gp.get_parameter_vector())
        y_pred_scaled, _ = self.gp.predict(self.y_train_scaled, X_pred_scaled)
        return self.unscale_y(y_pred_scaled)

    def callback(self, xk, *_):
        """Callback function for scipy.optimize.minimize.
        "*_" makes sure that it still works when the optimizer calls the callback function with more than one argument.
        xk = current parameter vector """
        xk = np.atleast_1d(xk)
        # search backwards in input list for input corresponding to xk
        for i, x in reversed(list(enumerate(self.inputs))):
            x = np.atleast_1d(x)
            if np.allclose(x, xk):
                break
        # if first callback, print labels
        if not self.callback_count:
            s0 = f"niter\t"
            for k in range(self.X_train.shape[1]):
                label = f"kparam-{k}"
                s0 += f"{label:8s}\t"
            colnames = ['lnlike', 'ncalls', 'time']
            for name in colnames:
                s0 += f"{name:8s}\t"
            print(s0, flush=True)
        # print current values
        s1 = f"{self.callback_count}\t"
        for comp in xk:
            s1 += f"{comp:8.6f}\t"  # parameter vector
        # likelihood, number of function (lnlike) calls, and time for this iteration
        s1 += f"{self.lnlikes[i]:8.6f}\t" + f"{i:8d}\t" + datetime.now().strftime("%H:%M:%S")
        print(s1, flush=True)
        self.callback_count += 1


if __name__=='__main__':
    main()
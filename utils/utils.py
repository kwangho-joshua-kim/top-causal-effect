import numpy as np
import pandas as pd
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import FourierBasis
from skfda.ml.regression import LinearRegression


def numerical_integration(f, tseq):
    f_right = f[1:]
    f_left = f[:-1]
    delta_t = tseq[1:] - tseq[:-1]
    integral = np.sum((f_right+f_left)/2 * delta_t)
    return integral


def fit_functional_regression(sample, tseq, n_basis):
    """_summary_

    Args:
        sample (tuple or list): Sample used for estimation. Triplet of (phi, A, X).
            - phi: Collection of silhouette functions. Shape: [n, n_hom_dim, resolution].
            - A: Treatment. Shape: [n,].
            - X: Covariates of dimension d. Shape: [n, d].
        tseq (_type_): _description_
        n_basis (_type_): _description_

    Returns:
        _type_: _description_
    """
    phi, A, X = sample
    n_hom_dim = phi.shape[-2]   # number of homology dimensions

    # split training sample into control and treated group
    ind = A.astype(bool)    # indicator of treated individuals in train sample
    X0 = X[~ind]            # covariates of control group in train sample
    X1 = X[ind]             # covariates of treated group in train sample

    fb = FourierBasis(tseq[[0, -1]], n_basis)   # basis used for function-on-scalar regression
    estimators = []
    for hom_dim in range(n_hom_dim):
        # control group
        phi0 = phi[~ind, hom_dim, :]    # silhouette function in "hom_dim"-dimesion for control group.
        phi0_fd = FDataGrid(phi0, tseq)  # representation of functional data as a set of curves discretised in a grid of points
        phi0_fb = phi0_fd.to_basis(fb)   # functional data form
        f_reg0 = LinearRegression().fit(pd.DataFrame(X0), phi0_fb)   # fit function-on-scalar regression

        # treated group
        phi1 = phi[ind, hom_dim, :]     # silhouette function in "hom_dim"-dimesion for treated group.
        phi1_fd = FDataGrid(phi1, tseq)
        phi1_fb = phi1_fd.to_basis(fb)
        f_reg1 = LinearRegression().fit(pd.DataFrame(X1), phi1_fb)

        estimators.append((f_reg0, f_reg1))
    return estimators
import numpy as np
import pandas as pd


def ipw_estimator(pi_hat, sample, return_inv_weight=False):
    """_summary_

    Args:
        pi_hat (np.array of shape (n,)): Estimated propensity score.
        sample (tuple or list): Sample used for estimation. Triplet of (phi, A, X).
            - phi: Collection of silhouette functions. Shape: [n, n_hom_dim, resolution].
            - A: Treatment. Shape: [n,].
            - X: Covariates of dimension d. Shape: [n, d].
        return_inv_weight (bool, optional): Wheter to return inverse weight values. Defaults to False.

    Returns:
        (list): List containing "n_hom_dim" IPW estimates of shape [resolution, ].
    """
    phi, A, _ = sample
    n_hom_dim = phi.shape[-2]   # number of homology dimensions
    
    # avoid 0 or 1 estimated propensity score
    pi_hat[pi_hat == 0] = 1e-2   
    pi_hat[pi_hat == 1] = 1-1e-2

    # construct ipw estimator on estimation sample
    inv_weight = (A/pi_hat - (1-A)/(1-pi_hat))[:, np.newaxis]   # shape: [n, 1] 
    
    ipw = []
    for hom_dim in range(n_hom_dim):
        ipw.append(np.mean(inv_weight * phi[:, hom_dim, :], axis=0))
    
    if return_inv_weight:
        return ipw, inv_weight
    return ipw


def plugin_estimator(mu_hats, return_mu=False):
    """_summary_

    Args:
        mu_hats (list of length "n_hom_dim"): Each element is a tuple (predicted mu_hat0, predicted mu_hat1).
        return_mu (bool, optional): Wheter to return regression function values. Defaults to False.

    Returns:
        (list): List containing "n_hom_dim" plug-in estimates of shape [resolution, ].
    """    
    plugin = [np.mean(mu1 - mu0, axis=0) for mu0, mu1 in mu_hats]

    if return_mu:
        mu0_list, mu1_list = zip(*mu_hats)
        return plugin, mu0_list, mu1_list
    return plugin


def aipw_estimator(pi_hat, mu_hats, sample):
    """_summary_

    Args:
        pi_hat (np.array of shape (n,)): Estimated propensity score.
        mu_hats (list of length "n_hom_dim"): Each element is a tuple (predicted mu_hat0, predicted mu_hat1).
        sample (tuple or list): Sample used for estimation. Triplet of (phi, A, X).
            - phi_est: Collection of silhouette functions. Shape: [n, n_hom_dim, resolution].
            - A_est: Treatment. Shape: [n,].
            - X_est: Covariates of dimension d. Shape: [n, d].
    
    Returns:
        (list): List containing "n_hom_dim" doubly robust estimates of shape [resolution, ].
    """
    phi, A, _ = sample
    n_hom_dim = phi.shape[-2]       # number of homology dimensions

    ipw, inv_weight = ipw_estimator(pi_hat, sample, return_inv_weight=True)
    plugin, mu0_list, mu1_list = plugin_estimator(mu_hats, return_mu=True)

    dr =[]
    A = A[:, np.newaxis]
    for hom_dim in range(n_hom_dim):
        correction = ipw[hom_dim] - np.mean(inv_weight*(A*mu1_list[hom_dim] + (1-A)*mu0_list[hom_dim]), axis=0)
        dr.append(plugin[hom_dim] + correction)
    return dr



import numpy as np


def gen_covariate(mu1, mu2, Sigma, n):
    """Data generation process. Returns covariates of each subgroup and the entire dataset.

    Args:
        mu1 (_type_): _description_
        mu2 (_type_): _description_
        Sigma (_type_): _description_
        n (_type_): _description_
    """
    size = int(n/2)
    covariate1 = np.random.multivariate_normal(mean=mu1, cov=Sigma, size=size)      # subgroup 1
    covariate2 = np.random.multivariate_normal(mean=mu2, cov=Sigma, size=(n-size))  # subgroup 2
    covariate = np.concat([covariate1, covariate2], axis=0)
    return covariate1, covariate2, covariate


def gen_trt_prob(covariate, beta):
    """Treatment mechanism. Returns probability of being assigned treatment and the treatment assignment.

    Args:
        covariate (_type_): _description_
        beta (_type_): _description_

    Returns:
        _type_: _description_
    """
    y = covariate @ beta + 0.5*covariate[:, 1]*covariate[:, 2] -0.7*covariate[:, 0]*covariate[:, 2]
    prob = 1 / (1 + np.exp(-y))
    A = np.random.binomial(n=1, p=prob)
    return prob, A
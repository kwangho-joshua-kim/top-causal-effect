import numpy as np
from gudhi.representations import Silhouette


def power_weight(point, r=3):
    birth, death = point
    return np.abs(death - birth)**r


def compute_silhouette(diags, interval=[0, 0.2], r=3, resolution=100):
    """_summary_

    Args:
        diags (list of n x 2 numpy arrays): List containing persistence diagrams
        interval (list, optional): _description_. Defaults to [0, 0.2].
        r (int, optional): _description_. Defaults to 3.
        res (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """

    silhouette = Silhouette(weight=lambda x: power_weight(x, r), resolution=resolution, sample_range=interval, keep_endpoints=True)
    return silhouette.fit_transform(diags)
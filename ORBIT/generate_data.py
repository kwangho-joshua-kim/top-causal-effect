import sys
sys.path.append("../")
import torch
import os


def gen_orbit(num_pts, rho):
    """Generate one orbit.

    Args:
        num_pts (int): Number of points in one orbit.
        rho (float): Parameter defining the dynamical system.

    Returns:
        torch.Tensor: Tensor of shape [num_pts, 2]
    """
    X = torch.zeros(num_pts, 2)
    x, y = torch.rand(1).item(), torch.rand(1).item()
    for i in range(num_pts):
        x = (x + rho * y * (1-y)) % 1
        y = (y + rho * x * (1-x)) % 1
        X[i] = torch.tensor([x, y])
    return X


def gen_orbits(rhos=[2.5, 3.5, 4.0, 4.1, 4.3], num_pts=1000, num_orbits_each=1000):
    """Generate entire ORBIT dataset.

    Args:
        rhos (list, optional): List of parameters defining the dynamical system. Defaults to [2.5, 3.5, 4.0, 4.1, 4.3].
        num_pts (int, optional): Number of points in one orbit. Defaults to 1000.
        num_orbits_each (int, optional): Number of data for each label. Defaults to 1000.

    Returns:
        _type_: _description_
    """
    X = torch.zeros(len(rhos)*num_orbits_each, num_pts, 2)
    y = []
    for label, rho in enumerate(rhos):
        for i in range(num_orbits_each):
            X[label*num_orbits_each + i] = gen_orbit(num_pts, rho)
            y.append(label)
    return X, torch.tensor(y)


if __name__ == "__main__":
    num_orbits_each = 1000                              # sample (per label) sizes
    dir = "./data"
    os.makedirs(dir, exist_ok=True)
    torch.manual_seed(123)

    X, y = gen_orbits(num_orbits_each=num_orbits_each)
    torch.save((X, y), f=f"{dir}/data.pt")
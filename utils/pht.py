import torch
import numpy as np
import gudhi as gd


def pht(graph, v, max_filtration=1.):
    """_summary_

    Args:
        graph (pyg Data): Graph in the format of pytorch_geometric Data type.
        v (torch.Tensor): Direction.
        max_filtration (float, optional): Max filtration value. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # node height
    node_h = graph.pos @ v                              # shape: [n_nodes, ]
    # edge height
    edge_h, _ = node_h[graph.edge_index].max(dim=0)     # shape: [n_edges, ]
    
    # build filtration
    n_nodes = graph.num_nodes
    filtration = torch.full((n_nodes, n_nodes), torch.inf)
    filtration[range(n_nodes), range(n_nodes)] = node_h             # insert node filtration values
    filtration[graph.edge_index[0], graph.edge_index[1]] = edge_h   # insert edge filtration values

    # compute ph
    st = gd.SimplexTree()
    st = st.create_from_array(filtration, max_filtration=max_filtration)
    st.compute_persistence(persistence_dim_max=True)

    diags = []
    for hom_dim in range(len(v)):
        pd = st.persistence_intervals_in_dimension(hom_dim)
        if hom_dim == 0:    # remove infinite filtation value
            pd = pd[~np.isinf(pd).any(axis=1)]
        else:               # replace infinite filtration value with max_filtration 
            pd[np.isinf(pd)] = max_filtration
        diags.append(pd)
    return diags


def sample_directions(n_samples: int, d: int) -> np.ndarray:
    """
    Deterministically sample points on the (d-1)-sphere S^{d-1} using generalized spherical coordinates.

    Parameters:
        n_samples (int): Number of samples to generate.
        d (int): Dimension of the ambient space R^d (so the sphere is S^{d-1}).

    Returns:
        np.ndarray: (n_samples, d) array of points on the unit sphere.
    """
    if d < 2:
        raise ValueError("Dimension must be at least 2 to define a sphere.")

    # Create linspace angles
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    
    samples = []

    for theta in angles:
        coords = [np.cos(theta)]  # x1 = cos(theta)
        sin_prod = np.sin(theta)  # sin(theta)

        # Use fixed angles for the rest, e.g., π/4 (could be made more sophisticated)
        for i in range(1, d - 1):
            phi = np.pi / 4
            coords.append(sin_prod * np.cos(phi))
            sin_prod *= np.sin(phi)

        coords.append(sin_prod)  # last coordinate
        samples.append(coords)

    return torch.Tensor(samples).to(torch.float32)
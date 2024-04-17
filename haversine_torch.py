import torch
import gc
from typing import List, Tuple

def haversine_torch(
    left: List[List[float]] | List[Tuple[float, float]], 
    right: List[List[float]] | List[Tuple[float, float]], 
    matrix: bool = True,
    R: int = 6371000,
):
    """
    The function calculates the pairwise distances between point in two lists in meters with >99.5% accuracy. 
    PyTorch vectorized implementation allows usage in CPU and GPU.
    Wall time ~10.5s for 1.1bn pairs - test sample 8200*132000
    
    Example
    -------
    
    l = [[1.01, 1.01], [2.01, 2.01]]
    r = [[2.01, 2.01], [3.01, 3.01]]
    dist_martix = haversine_torch(l,r)
    
    Parameters
    ----------
    
    left: List[List[float]] or List[Tuple[float]]
        List of lists or tuples with coordinates with [latitude, longitude]
    right: List[List[float]] or List[Tuple[float]]
        List of lists or tuples with coordinates with [latitude, longitude]
    matrix: bool
        If you want to receive the matrix of left*right or tensor of length left*right. Default True.
    R: int
        Radius of planet where you wnat to measure distance. Default value = 6371000m as radius of earth.
    """

    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'
        
    left_tensor = torch.tensor(left, dtype=torch.float32)
    right_tensor = torch.tensor(right, dtype=torch.float32)
    
    lat1, lon1 = torch.split(left_tensor, 1, dim=1)
    lat2, lon2 = torch.split(right_tensor, 1, dim=1)

    phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    lamb1, lamb2 = torch.deg2rad(lon1), torch.deg2rad(lon2)

    lat_cart = torch.cartesian_prod(torch.squeeze(phi1), torch.squeeze(phi2))
    lat_cart1, lat_cart2 = torch.split(lat_cart, 1, dim=1)

    sin_lat1, cos_lat1 = torch.sin(lat_cart1), torch.cos(lat_cart1)
    sin_lat2, cos_lat2 = torch.sin(lat_cart2), torch.cos(lat_cart2)
    
    lamb_dist = torch.reshape(torch.cdist(lamb1, lamb2, p=1), [len(lamb1)*len(lamb2), 1])
    cos_lamb_dist, sin_lamb_dist = torch.cos(lamb_dist), torch.sin(lamb_dist)

    dists = R*torch.atan2(torch.sqrt((cos_lat2 * sin_lamb_dist) ** 2 +
                       (cos_lat1 * sin_lat2 -
                        sin_lat1 * cos_lat2 * cos_lamb_dist) ** 2),
                        sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_lamb_dist)
    gc.collect()
    if matrix:
        dists = torch.reshape(dists, [len(lamb1), len(lamb2)])
    
    return dists
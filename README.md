# haversine_pytorch
The pairwise distance calculation between two sets of points with Haversine formula for CPU/GPU vectorized in PyTorch.

The function calculates the pairwise distances between point in two lists in meters with >99.5% accuracy. 
PyTorch vectorized implementation allows usage in CPU and GPU.

Comaprison
----------

haversine_torch
    Wall time ~5.9s:
        - CPU 
        - 568m pairs
        - test sample 8,154*69,600

    Wall time ~10.7s:
         - CPU
         - 1.1b
         - test sample 8,154*132,000


sklearn.metrics.pairwise.haversine_distances
    Wall time ~37.4s: 
         - CPU
         - 568m pairs
         - test sample 8,154*69,600


    
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
        Radius of planet where you wnat to measure distance. Default value = 6371000 according to the radius of the Earth.

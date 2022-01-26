import numpy as np
import numpy.typing as npt
from typing import Tuple,Sequence

def project3dPtsTo2d(points3d : npt.NDArray, Rtilt : npt.NDArray, crop : npt.ArrayLike, K : npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    # %% inverse of get_aligned_point_cloud
    points3d =(Rtilt.T @ points3d.T).T
    
    # %% inverse rgb_plane2rgb_world
    
    
    # % Now, swap Y and Z.
    points3d[:, [1, 2]] = points3d[:,[2, 1]]
    
    # % Make the original consistent with the camera location:
    x3 =  points3d[:,0]
    y3 = -points3d[:,1]
    z3 =  points3d[:,2]
    
    xx = x3 * K[0,0] / z3 + K[0,2]
    yy = y3 * K[1,1] / z3 + K[1,2]

    xx = xx - crop[1] + 1 #for one origin?
    yy = yy - crop[0] + 1

    points2d = np.hstack([xx, yy])

    return points2d, z3

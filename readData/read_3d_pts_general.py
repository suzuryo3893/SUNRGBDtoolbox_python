import os
from PIL import Image
import numpy as np
import numpy.typing as npt
from typing import Sequence, Optional, Tuple

#def read_3d_pts_general(depthInpaint, K, depthInpaintsize, imageName, crop):
def read_3d_pts_general(depthInpaint : npt.NDArray, K : npt.NDArray, depthInpaintsize : Sequence[int], imageName : Optional[str] = None) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64],npt.NDArray[np.float64]]:
    """
    Convert depth image to 3D points using camera parameter K.
    It also outputs color of each points if imageName is provided.
    % K is [fx 0 cx; 0 fy cy; 0 0 1];  
    % for uncrop image crop =[1,1];
    % imageName is the full path to image
    """
    cx : float = K[0,2]
    cy : float = K[1,2]
    fx : float = K[0,0]
    fy : float = K[1,1]
    invalid : npt.NDArray[np.bool_] = depthInpaint==0
    rgb : npt.NDArray[np.float64]
    if imageName is not None:
        #im = cv2.imread(imageName, cv2.IMREAD_COLOR)
        with Image.open(imageName) as _i :
            im = np.array(_i)
        rgb = im.astype(np.float64)/255.0
    else:
        rgb = np.dstack((
            np.zeros(depthInpaintsize, dtype=np.float64),
            np.ones(depthInpaintsize, dtype=np.float64),
            np.zeros(depthInpaintsize, dtype=np.float64)))
    rgb = rgb.reshape(-1, 3)
    # %3D points
    #x,y = np.meshgrid(range(0,depthInpaintsize[1]), range(0,depthInpaintsize[0]))
    x,y = np.meshgrid(range(1,depthInpaintsize[1]+1), range(1,depthInpaintsize[0]+1))
    x3 = (x.astype(np.float64)-cx)*depthInpaint*1/fx
    y3 = (y.astype(np.float64)-cy)*depthInpaint*1/fy
    z3 = depthInpaint.astype(np.float64)
    points3dMatrix = np.dstack((x3,z3,-y3))
    points3dMatrix[np.dstack((invalid,invalid,invalid))] = np.nan
    points3d = np.dstack([x3[:], z3[:], -y3[:]]).reshape(-1,3)
    points3d[invalid.reshape(-1),:] = np.nan

    return rgb, points3d, points3dMatrix
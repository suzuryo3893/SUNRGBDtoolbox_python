import os
from PIL import Image
import numpy as np
import numpy.typing as npt
from typing import Sequence, Tuple
from .read_3d_pts_general import read_3d_pts_general

def read3dPoints(data : npt.NDArray, data_root : str = '', datapath_replace :str = '/n/fs/sun3d/data/') -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], Sequence[int]]:
    """
    Read and convert data as 3D points
    data : data in SUNRGBD dataset
    data_root : root directory for SUNRGBD dataset
    datapath_replace : part of path written in SUNRGBD file path to replace with data_root
    """
    if len(data_root)>0:
        depthpath=os.path.join(data_root,data['depthpath'][0][len(datapath_replace):])
        rgbpath=os.path.join(data_root,data['rgbpath'][0][len(datapath_replace):])
    with Image.open(depthpath) as _i:
        depthVis = np.array(_i).astype(np.uint16)
    imsize = depthVis.shape
    _depthInpaint = depthVis >> 3 | depthVis << (16-3)
    depthInpaint = _depthInpaint.astype(np.float32)/1000
    depthInpaint[depthInpaint > 8] = 8
    rgb, points3d, _ = read_3d_pts_general(depthInpaint, data['K'], depthInpaint.shape, rgbpath)
    points3d = (data['Rtilt']@points3d.T).T

    return rgb, points3d, depthInpaint, imsize

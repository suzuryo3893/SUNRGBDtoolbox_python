import numpy as np
import numpy.typing as npt
from typing import Tuple,Dict,Optional

from .get_corners_of_bb3d import get_corners_of_bb3d
from .project3dPtsTo2d import project3dPtsTo2d

def projectStructBbsTo2d(bb : Dict = {}, Rtilt : npt.NDArray = np.array([]), crop : npt.ArrayLike = [1,1], K : npt.NDArray = np.array([])) -> Tuple[npt.NDArray,npt.NDArray]:
    if len(bb)==0:
        bb2d = np.array([])
        bb2dDraw = np.array([])
    else:
        nBbs = len(bb)
        if 'confidence' in bb:
            conf = bb['confidence']
        else:
            conf = np.ones((nBbs,1))
        
        points3d = np.zeros((8*nBbs, 3))
        bbk=list(bb.keys())
        for i in range(nBbs):
            corners = get_corners_of_bb3d(bb[bbk[i]])
            points3d[range(i*8,(i+1)*8), :] = corners[[7, 3, 4, 0, 6, 2, 5, 1], :]
            # %points3d((i-1)*8+(1:8),:) = corners([5,1,8,4,6,2,3,7],:);
        
        points2d,_ = project3dPtsTo2d(points3d, Rtilt, crop, K)

        bb2d = np.zeros((nBbs,5))
        bb2d[:,0] = np.min(points2d[:,0].reshape([8,nBbs]), axis=0)
        bb2d[:,1] = np.min(points2d[:,1].reshape([8,nBbs]), axis=0)
        bb2d[:,2] = np.max(points2d[:,0].reshape([8,nBbs]), axis=0)
        bb2d[:,3] = np.max(points2d[:,1].reshape([8,nBbs]), axis=0)
        bb2d[:,2] = bb2d[:,2] - bb2d[:,0]
        bb2d[:,3] = bb2d[:,3] - bb2d[:,1]
        bb2d[:,4] = conf

        bb2dDraw = np.zeros((nBbs,17))
        pts = points2d.T
        pts = pts.reshape([16,nBbs])
        bb2dDraw[:,0:16] = pts.T
        bb2dDraw[:,16] = conf
    
    return bb2d,bb2dDraw

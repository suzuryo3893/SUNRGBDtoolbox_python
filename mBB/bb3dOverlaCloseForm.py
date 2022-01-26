import numpy as np
import numpy.typing as npt
from .get_corners_of_bb3d import get_corners_of_bb3d
from .cuboidVolume import cuboidVolume
from .cuboidIntersectionVolume import cuboidIntersectionVolume

def bb3dOverlapCloseForm(bb1input : npt.NDArray, bb2struct: npt.NDArray) -> npt.NDArray:
    """
    % to run 
    % bb1 =[0,0,0,1,1,1];load('bb3d'); bb3dOverlapCloseForm(bb1,bb3d);
    % bb1 can be  nx6 bb or struct as bb2struct
    % convert BB to format : x1 y1 x2 y2 x3 y3 x4 y4 zMin zMax    
    """
    if bb1input.size==0 or bb2struct.size==0:
        return np.array([])
        
    nBb1 = bb1input.shape[0]
    nBb2 = bb2struct.shape[0]
    if bb1input.shape[1]>=6 and bb1input.shape[1]<10:
        bb1 = bb1input
        bb1[:, 3:6] = bb1input[:, 3:6] + bb1input[:,0:3]
        xMax = bb1[:,3]
        yMax = bb1[:,4]
        #bb1 = [bb1(:,1) bb1(:,2) xMax bb1(:,2) xMax yMax bb1(:,1) yMax bb1(:,3) bb1(:,6)];\
        bb1 = np.hstack([bb1[:,0], bb1[:,1], xMax, bb1[:,1], xMax, yMax, bb1[:,0], yMax, bb1[:,2], bb1[:,5]])
    elif bb1input.shape[1]==1:
        bb1 = np.zeros((nBb1,10))
        for i in range(nBb1):
            corners = get_corners_of_bb3d(bb1input[i])
            #bb1(i,:) = [reshape([corners(1:4,1) corners(1:4,2)]',1,[]) min(corners([1 end],3)) max(corners([1 end],3))];
            bb1[i,:] = np.hstack([
                corners[0:4,0],
                corners[0:4,1],
                min(corners[[1, -1], 3]),
                max(corners[[1, -1], 3])
            ])
    elif bb1input.shape[1] >= 10:
        bb1 = bb1input[:, 0:10]

    bb2 = np.zeros((nBb2, 10))
    for i in range(nBb2):
        corners = get_corners_of_bb3d(bb2struct[i])
        bb2[i,:] = np.hstack([
            corners[0:4,0],
            corners[0:4,1],
            min(corners[[1, -1], 3]),
            max(corners[[1, -1], 3])
        ])
    

    bb1 = bb1.T
    bb2 = bb2.T

    # % a ha, we are done with dirty format conversion


    nBb1 = bb1.shape[1]
    nBb2 = bb2.shape[1]

    volume1 = cuboidVolume(bb1)
    volume2 = cuboidVolume(bb2)
    intersection = cuboidIntersectionVolume(bb1.astype(np.float64),bb2.astype(np.float64))

    # %{
    # volume1(6818)
    # volume2(1)
    # intersection(6818,1)
    # cuboidVolume(bb1(:,6818))
    # cuboidVolume(bb2(:,1))
    # cuboidIntersectionVolume(bb1(:,6818),bb2(:,1))
    # cuboidDraw(bb1(:,6818))
    # cuboidDraw(bb2(:,1))
    # %}

    #union = repmat(volume1.T, 1,nBb2) + repmat(volume2,nBb1,1) - intersection
    union = np.hstack([volume1.T]*nBb2) + np.vstack([volume2]*nBb1) - intersection

    scoreMatrix = intersection / union
    
    return scoreMatrix
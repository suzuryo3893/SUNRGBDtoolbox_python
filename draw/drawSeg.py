import numpy as np
import cv2
import scipy.io
from .myObjectColor import myObjectColor
from ..readframeSUNRGBD import readframeSUNRGBD as readframe
from ..getSequenceName import getSequenceName
from .visualize_wholeroom import visualize_wholeroom

def drawSeg(instances, labels, image, GroundTruthBox):

    mask = 5*np.array(instances[:,:,1],dtype=np.float64) + 5*np.array(labels[:,:,1], dtype=np.float64)
    maskColor = np.zeros(image.shape[0]*image.shape[1], 3)
    uniquemask = list(set(mask))
    for i in range(len(uniquemask)):
        sel = mask==uniquemask(i)
        maskColor[sel, :] = np.fill(myObjectColor(uniquemask[i]), (sum([1 if x else 0 for x in sel]),1))

    maskColor[find(mask==0), :] = 1
    maskColor = maskColor.reshape([mask.shape[0], mask.shape[1], 3])

    # figure
    cv2.imshow("figure", maskColor)
    cv2.imwrite('NYUmask.png', maskColor)
    # %%
    # load('/n/fs/modelnet/SUN3DV2/prepareGT/cls.mat')
    cls = scipy.io.loadmat('/n/fs/modelnet/SUN3DV2/prepareGT/cls.mat')['cls']

    # addpath('/n/fs/modelnet/SUN3DV2/roomlayout/')
    fullname = '/n/fs/sun3d/data/rgbd_voc/000414_2014-06-04_19-49-13_260595134347_rgbf000044-resize'
    data = readframe(fullname)
    groundTruthBbs  = data['groundtruth3DBB']

    sequenceName = getSequenceName(fullname)
    gtRoom3D = GroundTruthBox[sequenceName, 0]
    cameraXYZ = data['anno_extrinsics'].T @ gtRoom3D
    cameraXYZ[[1, 2], :] = cameraXYZ[[2, 1],:]
    cameraXYZ[2,:] = -cameraXYZ[2,:]
    cameraXYZ = data.Rtilt @ cameraXYZ

    my_mhCorner3D = cameraXYZ.copy(); #%data.Rtilt*data.anno_extrinsics'*gtCorner3D;
    visualize_wholeroom(groundTruthBbs, cls, fullname, my_mhCorner3D)
    
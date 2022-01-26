# %% imports
import os
import numpy as np
from PIL import Image
import scipy.io
import h5py
from typing import Dict
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from readData.read3dPoints import read3dPoints
from draw.vis_point_cloud import vis_point_cloud
from draw.vis_cube import vis_cube
from draw.vis_line import vis_line

def loadSUNRGB(fname : str) -> Dict:
    try:
        _t = scipy.io.loadmat(fname)
    except NotImplementedError as e:
        _t = h5py.File(fname,'r')
    return _t

# %% variables
odir='/path/to/OriginalSUNRGBDToolbox'
ddir='/path/to/ORIGINAL_SUNRGBD'

colors=[(1,0,0),(0,1,0),(0,0,1),(0,1,1),(1,0,1),(1,1,0)]

SUNRGBDMeta = scipy.io.loadmat(os.path.join(odir,'Metadata/SUNRGBDMeta.mat'))['SUNRGBDMeta']
# SUNRGBD2Dseg created before MATLAB v7.0, and is hdf5 format
oSUNRGBD2Dseg = loadSUNRGB(os.path.join(odir,'Metadata/SUNRGBD2Dseg.mat'))
SUNRGBD2Dseg = oSUNRGBD2Dseg['SUNRGBD2Dseg']

imageId = 30
data = SUNRGBDMeta[0,imageId]
rgb, points3d, depthInpaint, imsize = read3dPoints(data,ddir)

# %% draw 
rgimg_disp = Image.fromarray((rgb.reshape((imsize[0],imsize[1],3))*255).astype(np.uint8))
rgimg_fig = plt.figure()
rgimg_ax = rgimg_fig.add_subplot(1,1,1)
rgimg_ax.imshow(rgimg_disp)
for kk in range(data['groundtruth3DBB'].shape[1]):
    bb=data['groundtruth3DBB'][0,kk]['gtBb2D'].astype(float)
    tl=bb[0,0:2]#xy
    br=tl+bb[0,2:4]#wh
    pts=[tl,[br[0],tl[1]],br,[tl[0],br[1]]]
    text = data['groundtruth3DBB'][0,kk]['classname'][0]
    linecolor=colors[kk%len(colors)]
    textcolor=(0,0,0)
    vis_line(rgimg_ax, pts[0], pts[1], linecolor, 2)
    vis_line(rgimg_ax, pts[1], pts[2], linecolor, 2)
    vis_line(rgimg_ax, pts[2], pts[3], linecolor, 2)
    vis_line(rgimg_ax, pts[3], pts[0], linecolor, 2)
    rgimg_ax.text(pts[0][0],pts[0][1],text, color=textcolor, backgroundcolor=linecolor)
plt.show()


# %% draw 3D 
pointcloud_fig = plt.figure()
pointcloud_ax = pointcloud_fig.add_subplot(1,1,1, projection="3d")
vis_point_cloud(pointcloud_ax, points3d, rgb)
# fid=open("bboxes.obj","w")
# f=1
for kk in range(data['groundtruth3DBB'].shape[1]):
    vis_cube(pointcloud_ax, data['groundtruth3DBB'][0,kk],(1.0, 0.0, 0.0))

    # from mBB.get_corners_of_bb3d import get_corners_of_bb3d
    # corners = get_corners_of_bb3d(data['groundtruth3DBB'][0,kk])
    # clsn = data['groundtruth3DBB'][0,kk]['classname'][0]
    # fid.write(f"o {clsn}\n")
    # for i in range(8):
    #     fid.write("v {:f} {:f} {:f}\n".format(*corners[i]))
    # fid.write("l {:d} {:d}\n".format(f+0,f+1))
    # fid.write("l {:d} {:d}\n".format(f+1,f+2))
    # fid.write("l {:d} {:d}\n".format(f+2,f+3))
    # fid.write("l {:d} {:d}\n".format(f+3,f+0))
    # fid.write("l {:d} {:d}\n".format(f+4,f+5))
    # fid.write("l {:d} {:d}\n".format(f+5,f+6))
    # fid.write("l {:d} {:d}\n".format(f+6,f+7))
    # fid.write("l {:d} {:d}\n".format(f+7,f+4))
    # fid.write("l {:d} {:d}\n".format(f+0,f+4))
    # fid.write("l {:d} {:d}\n".format(f+1,f+5))
    # fid.write("l {:d} {:d}\n".format(f+2,f+6))
    # fid.write("l {:d} {:d}\n".format(f+3,f+7))
    # f+=8
# fid.close()

# with open("pointcloud.pcd") as fid:
#     _p=points3d
#     v = np.logical_and.reduce(~np.isnan(_p),axis=1,initial=True)
#     p = _p[v,:]
#     c = (rgb[v,:]*255).astype(np.uint8)
#     npts=p.shape[0]
#     header=f"""
#     # .PCD v.7 - Point Cloud Data file format
#     VERSION .7
#     FIELDS x y z rgb
#     SIZE 4 4 4 4
#     TYPE F F F U
#     COUNT 1 1 1 1
#     WIDTH {npts}
#     HEIGHT 1
#     VIEWPOINT 0 0 0 1 0 0 0
#     POINTS {npts}
#     DATA ascii
#     """
#     fid.write(header)
#     for i in range(npts):
#         fid.write("{0:f} {1:f} {2:f} {3:d}\n".format(
#             p[i,0],
#             p[i,1],
#             p[i,2],
#             c[i,0]<<16 | c[i,1]<<8 | c[i,2]
#             ))
# %%
anno2d_label = oSUNRGBD2Dseg[SUNRGBD2Dseg["seglabel"][imageId,0]][()]
plt.imshow(anno2d_label/anno2d_label.max())

# % category name in 37 categories list
seg37list = scipy.io.loadmat(os.path.join(odir,'./Metadata/seg37list.mat'))['seg37list']
objectname37 = [seg37list[0,i][0] for i in range(seg37list.shape[1])]

seglabelall_label = oSUNRGBD2Dseg[SUNRGBD2Dseg["seglabelall"][imageId,0]][()]
plt.imshow(seglabelall_label/seglabelall_label.max())
# % category name of all categories
seglistall = scipy.io.loadmat(os.path.join(odir,'./Metadata/seglistall.mat'))['seglistall']
objectnameall = [seglistall[0,i][0] for i in range(seglistall.shape[1])]


# %% finalize
SUNRGBD2Dseg.close()
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict,List
import numpy.typing as npt

from vis_point_cloud import vis_point_cloud
from vis_cube import vis_cube
from drawRoom import drawRoom
from readframeSUNRGBD import readframeSUNRGBD as readframe
from readData.read3dPoints import read3dPoints
from myObjectColor import myObjectColor
from drawRoom import drawRoom

def visualize_wholeroom(BbsTight : Dict = None, cls : List = None, fullname : str = None, roomLayout : npt.NDArray = None, numofpoints2plot : int = 5000, savepath : str = None):
    sizeofpoint = max(round(200000/numofpoints2plot),3)
    Linewidth =5
    maxhight = 1.2
    vis = 'on'
    fig = plt.figure()
    ax = Axes3D(fig)
    # set(f, 'Position', [100, 100, 1149, 1249])
    if fullname is not None:
        data = readframe(fullname)
        rgb,points3d,_,imsize = read3dPoints(data)
        vis_point_cloud(ax, points3d, rgb, sizeofpoint, numofpoints2plot)
        maxhight = min(maxhight,max(points3d[:,2]))
    
    if BbsTight is not None:
        if cls is not None and not 'classid' in BbsTight:
            #_,classid = ismember({BbsTight.classname},cls)
            classid = cls.index(BbsTight['classname'])
        else:
            classid = BbsTight['classid']

        _btk=list(BbsTight.keys())
        for i in range(len(BbsTight)):
            vis_cube(ax, BbsTight[_btk[i]], myObjectColor(classid[i]), Linewidth)

    if roomLayout is not None:
        drawRoom(ax, roomLayout, 'b', Linewidth, maxhight)
    
    if savepath is not None:
        fig.savefig(savepath)

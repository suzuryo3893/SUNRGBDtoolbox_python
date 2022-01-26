import matplotlib.axes
import numpy as np
import numpy.typing as npt
from typing import Sequence
from .vis_line import vis_line

def drawRoom(ax : matplotlib.axes.Axes, roomLayout : npt.NDArray, color : Sequence, lineWidth : float ,maxhight : float):
    totalPoints  = roomLayout.shape[1]
    bottompoints= roomLayout[:, 0:totalPoints//2]
    toppoints = roomLayout[:, (totalPoints//2+1):totalPoints]
    toppoints[3, toppoints[3,:]>maxhight] = maxhight
    bottompoints[3, bottompoints[3,:]>maxhight] = maxhight
    ind = np.argmin(toppoints[2,:])
    for i in range(len(toppoints)-1):
        if i !=ind and i+1!=ind:
            vis_line(ax, toppoints[:,i].T, toppoints[:,i+1].T, color, lineWidth)
            vis_line(ax, bottompoints[:,i].T, bottompoints[:,i+1].T, color, lineWidth)
            vis_line(ax, toppoints[:,i].T, bottompoints[:,i].T, color, lineWidth)

    if 1!=ind and toppoints.shape[1]!=ind:
        vis_line(ax, toppoints[:,-1].T, toppoints[:,0].T, color, lineWidth)
        vis_line(ax, bottompoints[:,-1].T, bottompoints[:,0].T, color, lineWidth)
        vis_line(ax, toppoints[:,-1].T, bottompoints[:,-1].T, color, lineWidth)

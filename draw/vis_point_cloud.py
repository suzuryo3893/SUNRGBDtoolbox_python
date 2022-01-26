import matplotlib.axes
import math
import numpy as np
import numpy.typing as npt
from typing import Sequence,Optional

def vis_point_cloud(ax: matplotlib.axes.Axes, points : npt.NDArray, colors : npt.NDArray = np.array([]), sizes : Sequence = [], sampleSize : Optional[int] = None):
    """ 
    % Visualizes a 3D point cloud.
    %
    % Args:
    %   points3d - Nx3 or Nx2 point cloud where N is the number of points.
    %   colors - (optional) Nx3 vector of colors or Nx1 vector of values which which 
    %            be scaled for visualization.
    %   sizes - (optional) Nx1 vector of point sizes or a scalar value which is applied
    %           to every point in the cloud.
    %   sampleSize - (optional) the maximum number of points to show. Note that since matlab
    %                is slow, around 5000 points is a good sampleSize is practice.
    %
    % Author: Nathan Silberman (silberman@cs.nyu.edu)
    """
    # removeNaN = sum(isnan(points),2)>0;
    removeNaN = np.logical_and.reduce(~np.isnan(points), axis=1, initial=True)
    points = points[removeNaN,:]
    N, D = points.shape
    assert D == 2 or D == 3, 'points must be Nx2 or Nx3'

    if colors.size==0:
        norms = math.sqrt(np.sum(points*points, 1))
        colors = values2colors(norms)
    else:
        colors=colors[removeNaN,:]

    if len(sizes)==0:
        sizes = [10] * N
    elif len(sizes) == 1:
        sizes = [sizes] * N
    elif len(sizes) != N:
        raise Exception('sizes:size', 'len(sizes) must be N')

    if sampleSize is None:
        sampleSize = 5000

    # % Sample the points, colors and sizes.
    N = points.shape[0]
    if N > sampleSize:
        _seq = np.random.permutation(np.arange(N))
        seq = list(_seq[0:sampleSize])

        points = points[seq, :]
        colors = colors[seq, :]
        sizes = [sizes[i] for i in seq]

    if points.shape[1]==2:
        vis_2d(ax, points, colors, sizes)
    elif points.shape[1]==3:
        vis_3d(ax, points, colors, sizes)
    else:
        raise Exception('Points must be either 2 or 3d')

    # axis equal
    # %view(0,90);
    # %s  view(0,8);


def values2colors(values):
    values = scale_values(values)
    inds = math.ceil(values * 255) + 1
    h = colormap(jet(256))
    colors = h[inds, :]
    return colors

def scale_values(values):
    if len(values)>1:
        values = values - min(values)
        values = values / max(values)
    return values

def vis_3d(ax, points, colors, sizes):
    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]
    ax.scatter3D(X, Y, Z, s=sizes, c=colors)#, 'filled')

def vis_2d(ax, points, colors, sizes):
    X = points[:,0]
    Y = points[:,1]
    ax.scatter(X, Y, s=sizes, c=colors)#, 'filled')

    ax.set_xlabel('x')
    ax.set_ylabel('y')

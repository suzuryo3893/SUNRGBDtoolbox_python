import matplotlib.axes
import numpy.typing as npt
from typing import Sequence
from .vis_line import vis_line

def draw_square_3d(ax : matplotlib.axes.Axes, corners : npt.NDArray, color : Sequence[float] = (1.0, 0.0, 0.0), lineWidth : float = 0.5):
    """
    % Draws a square in 3D
    %
    % Args:
    %   corners - 8x2 matrix of 2d corners.
    %   color - matlab color code, a single character.
    %   lineWidth - the width of each line of the square.
    %
    % Author: Nathan Silberman (silberman@cs.nyu.edu)
    """
    vis_line(ax, corners[0,:], corners[1,:], color, lineWidth)
    vis_line(ax, corners[1,:], corners[2,:], color, lineWidth)
    vis_line(ax, corners[2,:], corners[3,:], color, lineWidth)
    vis_line(ax, corners[3,:], corners[0,:], color, lineWidth)

    vis_line(ax, corners[4,:], corners[5,:], color, lineWidth)
    vis_line(ax, corners[5,:], corners[6,:], color, lineWidth)
    vis_line(ax, corners[6,:], corners[7,:], color, lineWidth)
    vis_line(ax, corners[7,:], corners[4,:], color, lineWidth)

    vis_line(ax, corners[0,:], corners[4,:], color, lineWidth)
    vis_line(ax, corners[1,:], corners[5,:], color, lineWidth)
    vis_line(ax, corners[2,:], corners[6,:], color, lineWidth)
    vis_line(ax, corners[3,:], corners[7,:], color, lineWidth)


import matplotlib.axes
from typing import Sequence

def vis_line(ax : matplotlib.axes.Axes, p1 : Sequence, p2 : Sequence, color : Sequence[float] = (1.0, 0.0, 0.0), lineWidth : float = 0.5):
    """
    % Visualizes a line in 2D or 3D space
    % 
    % Args:
    %   p1 - 1x2 or 1x3 point
    %   p2 - 1x2 or 1x3 point
    %   color - matlab color code, a single character
    %   lineWidth - the width of the drawn line
    %
    % Author: Nathan Silberman (silberman@cs.nyu.edu)
    """
    # % Make sure theyre the same size.
    assert len(p1) == len(p2), 'Vectors are of different dimensions'

    if len(p1)==2:
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], linewidth=lineWidth, color=color)
    elif len(p1)==3:
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]], linewidth=lineWidth, color=color)
    else:
        raise Exception('vectors must be either 2 or 3 dimensional')
    


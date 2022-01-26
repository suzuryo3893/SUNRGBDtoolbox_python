import matplotlib.axes
from typing import Dict,Sequence
from .draw_square_3d import draw_square_3d
from mBB.get_corners_of_bb3d import get_corners_of_bb3d

def vis_cube(ax : matplotlib.axes.Axes, bb3d : Dict, color : Sequence[float], lineWidth : float = 0.5):
  """
  % Visualizes a 3D bounding box.
  %
  % Args:
  %   bb3d - 3D bounding box struct
  %   color - matlab color code, a single character
  %   lineWidth - the width of each line of the square
  %
  % See:
  %   create_bounding_box_3d.m
  %
  % Author:
  %   Nathan Silberman (silberman@cs.nyu.edu)
  """
  corners = get_corners_of_bb3d(bb3d)
  draw_square_3d(ax, corners, color, lineWidth)

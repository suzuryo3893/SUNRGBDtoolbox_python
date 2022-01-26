import matplotlib.axes
import numpy as np
from typing import Sequence
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
def plotcube(ax : matplotlib.axes.Axes, edges : Sequence = [10,56,100], origin : Sequence = [10,10,10], alpha : float = 0.7, clr : Sequence= [1,0,0]):
  """
  PLOTCUBE - Display a 3D-cube in the current axes

    PLOTCUBE(EDGES,ORIGIN,ALPHA,COLOR) displays a 3D-cube in the current axes
    with the following properties:
    * EDGES : 3-elements vector that defines the length of cube edges
    * ORIGIN: 3-elements vector that defines the start point of the cube
    * ALPHA : scalar that defines the transparency of the cube faces (from 0
              to 1)
    * COLOR : 3-elements vector that defines the faces color of the cube

  Example:
    >> plotcube([5 5 5],[ 2  2  2],.8,[1 0 0]);
    >> plotcube([5 5 5],[10 10 10],.8,[0 1 0]);
    >> plotcube([5 5 5],[20 20 20],.8,[0 0 1]);
  """
  XYZ = np.array([
      [[0, 0, 0, 0],  [0, 0, 1, 1],  [0, 1, 1, 0]],
      [[1, 1, 1, 1],  [0, 0, 1, 1],  [0, 1, 1, 0]],
      [[0, 1, 1, 0],  [0, 0, 0, 0],  [0, 0, 1, 1]],
      [[0, 1, 1, 0],  [1, 1, 1, 1],  [0, 0, 1, 1]],
      [[0, 1, 1, 0],  [0, 0, 1, 1],  [0, 0, 0, 0]],
      [[0, 1, 1, 0],  [0, 0, 1, 1],  [1, 1, 1, 1]]
  ])

  nedges=np.array([edges]*6)
  norigin=np.array([origin]*6)
  nXYZ=XYZ.copy()
  for i in range(4):
      nXYZ[:,:,i] = XYZ[:,:,i] * nedges + norigin

  ax.add_collection3d(Poly3DCollection(nXYZ[:,:,:3]))

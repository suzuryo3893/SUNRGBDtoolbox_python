import numpy as np
import numpy.typing as npt
from typing import Dict

def get_corners_of_bb3d(bb3d : Dict) -> npt.NDArray:
    """
    % Gets the 3D coordinates of the corners of a 3D bounding box.
    %
    % Args:
    %   bb3d - 3D bounding box struct.
    %
    % Returns:
    %   corners - 8x3 matrix of 3D coordinates.
    %
    % See:
    %   create_bounding_box_3d.m
    %
    % Author: Nathan Silberman (silberman@cs.nyu.edu)
    """
    corners = np.zeros((8, 3))

    # % Order the bases.
    #sort by x
    inds = np.argsort(np.abs(bb3d['basis'][:,0]))[::-1] #descend
    basis = bb3d['basis'][inds, :] # Mx3 (M=3)
    coeffs = bb3d['coeffs'][0,inds] # 3

    #sort by y
    inds = np.argsort(np.abs(basis[1:3,1]))[::-1] #descend
    if inds[0] == 1:
        # flip second vector and third vector
        basis[1:3,:] = np.flip(basis[1:3,:], axis=0)
        coeffs[1:3] = np.flip(coeffs[1:3])
    

    # % Now, we know the basis vectors are orders X, Y, Z. Next, flip the basis
    # % vectors towards the viewer.
    basis = flip_towards_viewer(basis, #3x3
        #repmat(bb3d['centroid'], [3, 1])
        np.tile(bb3d['centroid'], (3,1)) # 3x3
        )

    coeffs = np.abs(coeffs)

    corners[0,:] = -basis[0,:] * coeffs[0] +  basis[1,:] * coeffs[1] +  basis[2,:] * coeffs[2]
    corners[1,:] =  basis[0,:] * coeffs[0] +  basis[1,:] * coeffs[1] +  basis[2,:] * coeffs[2]
    corners[2,:] =  basis[0,:] * coeffs[0] + -basis[1,:] * coeffs[1] +  basis[2,:] * coeffs[2]
    corners[3,:] = -basis[0,:] * coeffs[0] + -basis[1,:] * coeffs[1] +  basis[2,:] * coeffs[2]

    corners[4,:] = -basis[0,:] * coeffs[0] +  basis[1,:] * coeffs[1] + -basis[2,:] * coeffs[2]
    corners[5,:] =  basis[0,:] * coeffs[0] +  basis[1,:] * coeffs[1] + -basis[2,:] * coeffs[2]
    corners[6,:] =  basis[0,:] * coeffs[0] + -basis[1,:] * coeffs[1] + -basis[2,:] * coeffs[2]
    corners[7,:] = -basis[0,:] * coeffs[0] + -basis[1,:] * coeffs[1] + -basis[2,:] * coeffs[2]

    #corners = corners + repmat(bb3d.centroid, [8 1]);
    corners = corners + np.tile(bb3d['centroid'], (8,1))

    return corners


def flip_towards_viewer(normals : npt.NDArray, points : npt.NDArray) -> npt.NDArray:
    # points = points ./ repmat(sqrt(sum(points.^2, 2)), [1, 3]);
    points = points / np.tile(np.sqrt(np.sum(points**2, axis=1)), (3,1)).T

    proj = np.sum(points * normals, axis=1)

    flip = proj > 0
    normals[flip, :] = -normals[flip, :]

    return normals

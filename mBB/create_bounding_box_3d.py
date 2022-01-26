import numpy as np
import numpy.typing as npt
from typing import Dict

def create_bounding_box_3d(basis2d : npt.NDArray, centroid : npt.NDArray, coeffs : npt.NDArray) -> Dict:
    """
    % Helper method for quickly creating bounding boxes.
    %
    % Args:
    %   basis2d - 2x2 matrix for the basis in the XY plane
    %   centroid - 1x3 vector for the 3D centroid of the bounding box.
    %   coeffs - 1x3 vector for the radii in each dimension (x, y, and z)
    %
    % Returns:
    %   bb - a bounding box struct.
    %
    % Author: Nathan Silberman (silberman@cs.nyu.edu)
    """
    assert(basis2d.shape == (2, 2))
    assert(centroid.size == 3)
    assert(coeffs.size == 3)

    centroid = centroid.reshape(-1)
    coeffs = coeffs.reshape(-1)

    bb = {}
    bb['basis'] = np.zeros((3,3))
    bb['basis'][2,:] = [0, 0, 1]
    bb['basis'][0:2,0:2] = basis2d

    bb['centroid'] = centroid
    bb['coeffs'] = coeffs
    # %   bb.volume = prod(2 * bb.coeffs);

    return bb

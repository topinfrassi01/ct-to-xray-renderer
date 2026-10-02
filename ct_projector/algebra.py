import numpy as np
from math import cos, sin


def matrix_transform(matrix:np.ndarray, coordinates:np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates)
    if coordinates.ndim == 3: 
        h_dim = list(coordinates.shape[:-1]) + [1]
        h_coords = np.concatenate((coordinates, np.ones(h_dim)), axis=-1)
        h_coords = (matrix @ h_coords.transpose(0,2,1)).transpose(0,2,1)
    else:
        if coordinates.ndim == 1:
            coordinates = coordinates[None, ...]

        h_coords = np.hstack((coordinates, np.ones((len(coordinates), 1))))
        h_coords = (matrix @ h_coords.T).T
    h_coords /= h_coords[..., 3:]
    return h_coords[..., :3]


def lines_parallel_planes_intersection(
    l_0:np.ndarray,
    l_dir:np.ndarray,
    l_length:float,
    p_0:np.ndarray,
    p_n:np.ndarray,
    offsets:np.ndarray
) -> np.ndarray:
    """
    Returns the "t" value of the parametric form of intersections between lines and parallel planes.

    Args:
        l_0: Origin of lines, of shape [N,3]
        l_dir: Unit-vector direction of lines [N,3]
        l_length: Length of the rays [N,]
        p_0: Origin of the initial plane [1,3]
        p_n: Normal of the initial plane [1,3]
        offsets: Array of offsets of size O to apply to p_0 in the direction of p_n to define parallel planes. 
                 Should include 0 for the initial plane
    Returns:
        Intersections for all lines to all parallel planes [N,O,3]
    """
    dot_prod_down = np.vecdot(l_dir, p_n)
    dot_prod_up = np.vecdot(p_0 - l_0, p_n)[..., None]

    with np.errstate(divide='ignore', invalid='ignore'):
        return (dot_prod_up + offsets[None, ...]) / (dot_prod_down[..., None] * l_length)


def rotation_matrix(*, pitch:float = 0.0, yaw:float = 0.0, roll:float = 0.0) -> np.ndarray:
    """
    Creates a rotation matrix using three angles
    Args:
        pitch: Rotation around the X axis in radian
        yaw: Rotation around the Y axis in radian
        roll: Rotation around the Z axis in radian
    Returns:
        A 4x4 rotation matrix
    """
    pitch_matrix = np.eye(3)
    pitch_matrix[1,1] = cos(pitch)
    pitch_matrix[1,2] = -sin(pitch)
    pitch_matrix[2,1] = sin(pitch)
    pitch_matrix[2,2] = cos(pitch)

    yaw_matrix = np.eye(3)
    yaw_matrix[0,0] = cos(yaw)
    yaw_matrix[0,2] = sin(yaw)
    yaw_matrix[2,0] = -sin(yaw)
    yaw_matrix[2,2] = cos(yaw)

    roll_matrix = np.eye(3)
    roll_matrix[0,0] = cos(roll)
    roll_matrix[0,1] = -sin(roll)
    roll_matrix[1,0] = sin(roll)
    roll_matrix[1,1] = cos(roll)

    rotation_matrix = np.eye(4)
    rotation_matrix[:3,:3] = roll_matrix @ yaw_matrix @ pitch_matrix
    return rotation_matrix


def unit_vector_projection(v:np.ndarray, pts:np.ndarray) -> np.ndarray:
    """
    Projects points on a vector of norm 1
    If norm isn't one, the projection won't be good.

    Args:
        v: Unit vector
        pts: Points to project
    Returns:
        Points projected on the unit vector
    """
    dot_vv = np.dot(v,v)
    dots = np.vecdot(v[None, ...], pts, keepdims=True)
    return (v * dots) / dot_vv
from math import pi

from common import Image
from algebra import *

import numpy as np

class CtProjector:
    def __init__(
        self,
        image:Image
    ):
        """
        Creates an instance of the CT projector.
        Args:
            image: CT image to use to render x-rays
        """
        self._image = image
    
    def _define_uvn_frame(
        self,
        distance_to_detector:float,
        pitch:float = 0.0,
        yaw:float = 0.0,
        roll:float = 0.0
    ):
        image_center = np.eye(4)
        image_center[:3,3] = self._image.xyz_center

        world_frame = np.eye(4)
        world_frame[:3,:3] = self._image.directions

        world_to_uvn = rotation_matrix(pitch=-pi/2)

        translation_to_distance = np.eye(4)
        translation_to_distance[2, 3] = -distance_to_detector
        
        return image_center @ world_to_uvn @ rotation_matrix(pitch=pitch, yaw=yaw, roll=roll) @ translation_to_distance @ world_frame

    def _define_projection_lines(
        self,
        uvn_frame:np.ndarray,
        pixel_resolution:float,
        image_size:tuple[int,int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        half_w = image_size[0] // 2
        half_h = image_size[1] // 2
        arr_w = np.arange(-half_w, half_w)
        arr_h = np.arange(-half_h, half_h)

        x_values, y_values = np.meshgrid(arr_w, arr_h)
        pixels_3d = np.vstack([x_values.reshape(-1), y_values.reshape(-1), np.zeros(len(x_values)*len(y_values))]).T
        pixels_3d *= pixel_resolution

        converted = matrix_transform(np.linalg.inv(uvn_frame), self._image.xyz_corners)

        points = unit_vector_projection(np.array([0,0,1], dtype=float), converted)
        dist = np.max(points[:,2])
        
        pixels_3d = matrix_transform(uvn_frame, pixels_3d + [0,0,dist])
        
        line_directions = pixels_3d - uvn_frame[:3, 3]
        line_lengths = np.linalg.norm(line_directions, axis=1, keepdims=True)
        line_directions /= line_lengths

        return pixels_3d, line_directions, line_lengths
        

    def _identify_alphas(self, line_origins, line_directions, line_lengths):
        offsets = [[self._image.spacing[i] * j for j in range(self._image.ct_array.shape[i] + 1)] for i in range(3)]
        starting_indices = np.cumsum([0] + [len(o) for o in offsets])[:-1]
        ending_indices = starting_indices + [len(o) - 1 for o in offsets]
        all_alphas = None
        image_origin, image_planes = self._image.intersection_planes()
        for i, plane_normal in enumerate(image_planes):
            plane_offsets = np.array(offsets[i])
            alphas = lines_parallel_planes_intersection(line_origins, line_directions, line_lengths, image_origin, plane_normal, plane_offsets)
            all_alphas = np.hstack((all_alphas, alphas)) if all_alphas is not None else alphas

        alpha_mins = np.clip(np.max(np.minimum(all_alphas[:, starting_indices], all_alphas[:, ending_indices]), axis=1), a_min=0.0, a_max=None)[..., None]
        alpha_maxs = np.clip(np.min(np.maximum(all_alphas[:, starting_indices], all_alphas[:, ending_indices]), axis=1), a_min=None, a_max=1.0)[..., None]

        all_alphas = np.clip(all_alphas, a_min=alpha_mins, a_max=alpha_maxs)
        all_alphas.sort(axis=1, kind="M")
        return all_alphas

    def _identify_rhos(self, rays_origin, line_directions, line_lengths, all_alphas):
        line_start_ends = (rays_origin[None,...] + line_directions[None,...] * line_lengths[None,...] * all_alphas[:,[0,-1]].T[...,None])
        ijk = matrix_transform(self._image.xyz_to_ijk_matrix, line_start_ends + line_directions*1e-8)
        begins_ijk = ijk[0]
        ends_ijk = ijk[1]
        spread_ijk = ends_ijk - begins_ijk
        return np.rint(begins_ijk[:, None, :] + spread_ijk[:, None, :] * all_alphas[..., None])[:,:-1,:].astype(int)

    def render_xray(
        self,
        distance_to_detector:float,
        pitch:float,
        yaw:float,
        roll:float,
        pixel_resolution:float,
        image_size:tuple[int,int]
    ) -> np.ndarray:
        """
        Renders an X-ray image from the input CT
        Args:
            distance_to_detector: Distance between the origin of the UVN frame and the back of the CT scan
            pitch: Pitch (U) rotation to apply to the UVN frame
            yaw: Yaw (V) rotation to apply to the UVN frame
            roll: Roll (N) rotation to apply to the UVN frame
            pixel_resolution: Pixel resolution for the output
            image_size: Final image size (w,h)
        Returns:
            A uint8 x-ray rendering of a CT
        """
        uvn_frame = self._define_uvn_frame(distance_to_detector, pitch, yaw, roll)
        rays_origin = uvn_frame[:3, 3][None, ...]
        _, line_directions, line_lengths = self._define_projection_lines(uvn_frame, pixel_resolution, image_size)
        
        all_alphas = self._identify_alphas(rays_origin, line_directions, line_lengths)
        ls = (np.abs(np.diff(all_alphas, axis=1)) * line_lengths)

        rhos = self._identify_rhos(rays_origin, line_directions, line_lengths, all_alphas)
        traversed_voxels = self._image.ct_array[rhos[:,:,0], rhos[:,:,1], rhos[:,:,2]]
        image = np.sum(traversed_voxels * ls, axis=-1, dtype=np.float64).reshape(image_size)
        image = ((image - np.min(image)) / np.ptp(image) * 255).astype(np.uint8)
        return image

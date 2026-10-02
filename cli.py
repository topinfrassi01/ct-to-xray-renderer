
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from ct_projector import CtProjector, Image

def main():
    arg_parser = ArgumentParser("Renders an X-ray image from a CT ")
    arg_parser.add_argument("path", type=Path, required=True, help="Path to the CT scan. Refer to SimpleITK.ImageFileReader for valid inputs")
    arg_parser.add_argument("--source-to-detector-distance", type=float, default=500.0, help="Distance between the origin of the UVN frame and the back of the CT scan")
    arg_parser.add_argument("--pitch", type=float, default=0.0, help="Pitch rotation to apply to the UVN frame")
    arg_parser.add_argument("--yaw", type=float, default=0.0, help="Yaw rotation to apply to the UVN frame")
    arg_parser.add_argument("--roll", type=float, default=0.0, help="Roll rotation to apply to the UVN frame")
    arg_parser.add_argument("--pixel-resolution", type=float, default=1.0, help="Pixel resolution for the output")
    arg_parser.add_argument("--image-size", nargs=2, type=int, help="Final image size (w,h)")
    arg_parser.add_argument("--output", type=Path, help="Path to the output X-ray image")
    args = arg_parser.parse_args()
    reader = sitk.ImageFileReader()
    reader.SetFileName(args.path)
    
    itk_img:sitk.Image = reader.Execute()
    size = itk_img.GetSize()
    directions = np.reshape(itk_img.GetDirection(), (3,3))

    image = Image(
        sitk.GetArrayFromImage(itk_img).transpose(2,1,0),
        np.array(itk_img.GetOrigin()),
        np.array(itk_img.GetSpacing()),
        directions
    )
    
    ct_projector = CtProjector(image)

    image = ct_projector.render_xray(
        args.source_to_detector_distance,
        args.pitch,
        args.yaw,
        args.roll,
        args.pixel_resolution,
        args.image_size)
    
    sitk.WriteImage(image, args.output)
if __name__ == "__main__":
    main()

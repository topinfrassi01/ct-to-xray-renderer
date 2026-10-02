You can find information on this code here : https://medium.com/@francis.toupin/rendering-x-ray-images-from-ct-scans-934b30b75490?postPublishedType=initial

But basically, we do X-ray rendering from CT scans using the exact radiological path as formulated by [Siddon et al.](https://www2.physik.uni-muenchen.de/lehre/vorlesungen/wise_23_24/Vorlesung_-Computational-methods-in-medical-physics/Material/_auth/Siddon1985RadiologicalPath.pdf)

You should know this code uses only CPU. While I've vectorized all operations, it is still pretty slow. I'd like to use pycuda at some point to make it faster, but I'm not sure I'll get there.

## Installing

You need CMake in order to pip install numba, apart from that you should be able to simply install the requirements

If you want to run the CLI you'll also have to install SimpleITK

## Example

![Example of rendering](https://github.com/topinfrassi01/ct-to-xray-renderer/blob/main/figures/example.png)

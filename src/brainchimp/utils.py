"""
Ref
    - https://docs.pyvista.org/api/readers/_autosummary/pyvista.niftireader
    - https://www.analyticsvidhya.com/blog/2024/07/how-to-use-pyvista-for-interactive-3d-medical-visualizations/
"""

import nibabel as nib
import nibabel.processing
import numpy as np
import pyvista as pv


def print_img(img):
    data = img.get_fdata()

    print()
    print("shape            : ", data.shape)
    print("orientation      : ", nib.aff2axcodes(img.affine))
    print("voxel_size [mm]  : ", img.header.get_zooms())
    print("data type        : ", data.dtype)
    print("range            : ", data.min(), "->", data.max())
    print(img.affine)


def plot_img(img):
    if isinstance(img, str):
        data = pv.get_reader(img).read()
    else:
        arr = np.asarray(img.get_fdata(), dtype=np.float32)
        data = pv.ImageData(dimensions=arr.shape, spacing=img.header.get_zooms()[:3])
        data.point_data["values"] = arr.flatten(order="F")

    pl = pv.Plotter()
    pl.add_volume(data, cmap="gray", opacity="sigmoid", shade=True)
    pl.set_background("black")
    pl.show()

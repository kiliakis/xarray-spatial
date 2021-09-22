import numpy as np
import xarray as xr
import dask.array as da
import cupy as cp

from xrspatial.utils import has_cuda
#from xrspatial.utils import doesnt_have_cuda
from xrspatial.perlin import perlin

# import matplotlib.pyplot as plt

def create_test_arr(backend='numpy'):
    W = 20
    H = 30
    data = np.zeros((H, W), dtype=np.float32)
    raster = xr.DataArray(data, dims=['y', 'x'])

    if has_cuda() and 'cupy' in backend:
        raster.data = cp.asarray(raster.data)

    if 'dask' in backend:
        raster.data = da.from_array(raster.data, chunks=(10, 10))

    return raster


# vanilla numpy version
data_numpy = create_test_arr()
perlin_numpy = perlin(data_numpy)
# perlin_numpy.plot.imshow()
# im_numpy = plt.imshow(perlin_numpy)


# cupy
data_cupy = create_test_arr(backend='cupy')
perlin_cupy = perlin(data_cupy)
# im_cupy = plt.imshow()
assert np.isclose(
        perlin_numpy.data, perlin_cupy.data, equal_nan=True).all()


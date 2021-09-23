# std lib
from functools import partial

# 3rd-party
import numpy as np
import xarray as xr

try:
    import cupy
except ImportError:
    class cupy(object):
        ndarray = False

import dask.array as da
from numba import cuda, jit
import numba as nb

# local modules
from xrspatial.utils import has_cuda
from xrspatial.utils import cuda_args


@jit(nopython=True, nogil=True, parallel=True, cache=True)
def _lerp(a, b, x):
    return a + x * (b - a)


@jit(nopython=True, nogil=True, parallel=True, cache=True)
def _fade(t):
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


@jit(nopython=True, nogil=True, parallel=True, cache=True)
def _gradient(h, x, y):
    # assert(len(h.shape) == 2)
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    out = np.zeros(h.shape)
    for j in nb.prange(h.shape[1]):
        for i in nb.prange(h.shape[0]):
            f = np.mod(h[i, j], 4)
            g = vectors[f]
            out[i, j] = g[0] * x[i, j] + g[1] * y[i, j]
    return out


def _perlin(p, x, y):
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)

    # internal coordinates
    xf = x - xi
    yf = y - yi

    # fade factors
    u = _fade(xf)
    v = _fade(yf)

    # noise components
    n00 = _gradient(p[p[xi] + yi], xf, yf)
    n01 = _gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = _gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = _gradient(p[p[xi + 1] + yi], xf - 1, yf)

    # combine noises
    x1 = _lerp(n00, n10, u)
    x2 = _lerp(n01, n11, u)
    a = _lerp(x1, x2, v)
    return a


def _perlin_numpy(data: np.ndarray,
                  freq: tuple,
                  seed: int) -> np.ndarray:
    np.random.seed(seed)
    p = np.random.permutation(2**20)
    p = np.append(p, p)

    height, width = data.shape
    linx = np.linspace(0, freq[0], width, endpoint=False, dtype=np.float32)
    liny = np.linspace(0, freq[1], height, endpoint=False, dtype=np.float32)
    x, y = np.meshgrid(linx, liny)

    data[:] = _perlin(p, x, y)
    data[:] = (data - np.min(data)) / np.ptp(data)
    return data


def _perlin_dask_numpy(data: da.Array,
                       freq: tuple,
                       seed: int) -> da.Array:
    np.random.seed(seed)
    p = np.random.permutation(2**20)
    p = np.append(p, p)

    height, width = data.shape
    linx = da.linspace(0, freq[0], width, endpoint=False, dtype=np.float32)
    liny = da.linspace(0, freq[1], height, endpoint=False, dtype=np.float32)
    x, y = da.meshgrid(linx, liny)

    _func = partial(_perlin, p)
    data = da.map_blocks(_func, x, y, meta=np.array((), dtype=np.float32))

    data = (data - da.min(data)) / da.ptp(data)
    return data


@cuda.jit(device=True)
def _lerp_gpu(a, b, x):
    return a + x * (b - a)


@cuda.jit(device=True)
def _fade_gpu(t):
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


@cuda.jit(device=True)
def _gradient_gpu(vec, h, x, y):
    f = h % 4
    return vec[f][0] * x + vec[f][1] * y


@cuda.jit(fastmath=True, opt=True)
def _perlin_gpu(p, x0, x1, y0, y1, m, out):
    # alloc and initialize array to be used in the gradient routine
    vec = cuda.local.array((4, 2), nb.int32)
    vec[0][0] = vec[1][0] = vec[2][1] = vec[3][1] = 0
    vec[0][1] = vec[2][0] = 1
    vec[1][1] = vec[3][0] = -1

    i, j = nb.cuda.grid(2)
    if i < out.shape[0] and j < out.shape[1]:
        # coordinates of the top-left
        y = y0 + i * (y1 - y0) / out.shape[0]
        x = x0 + j * (x1 - x0) / out.shape[1]

        # coordinates of the top-left
        x_int = int(x)
        y_int = int(y)

        # internal coordinates
        xf = x - x_int
        yf = y - y_int

        # fade factors
        u = _fade_gpu(xf)
        v = _fade_gpu(yf)

        # noise components
        n00 = _gradient_gpu(vec, p[p[x_int] + y_int], xf, yf)
        n01 = _gradient_gpu(vec, p[p[x_int] + y_int + 1], xf, yf - 1)
        n11 = _gradient_gpu(vec, p[p[x_int + 1] + y_int + 1], xf - 1,
                                    yf - 1)
        n10 = _gradient_gpu(vec, p[p[x_int + 1] + y_int], xf - 1, yf)

        # combine noises
        x1 = _lerp_gpu(n00, n10, u)
        x2 = _lerp_gpu(n01, n11, u)
        out[i, j] = m * _lerp_gpu(x1, x2, v)
    


def _perlin_cupy(data: cupy.ndarray,
                 freq: tuple,
                 seed: int) -> cupy.ndarray:
    cupy.random.seed(seed)
    p = cupy.random.permutation(2**20)
    # np.random.seed(seed)
    # p = cupy.asarray(np.random.permutation(2**20))
    p = cupy.append(p, p)

    griddim, blockdim = cuda_args(data.shape)

    # blockdim = (24, 24)
    # griddim = tuple(int(d / blockdim[0] + 0.5) for d in data.shape)
    _perlin_gpu[griddim, blockdim](p, 0, freq[0], 0, freq[1], 1, data)

    minimum = cupy.amin(data)
    maximum = cupy.amax(data)
    data[:] = (data - minimum) / (maximum - minimum)
    return data


def perlin(agg: xr.DataArray,
           freq: tuple = (1, 1),
           seed: int = 5,
           name: str = 'perlin') -> xr.DataArray:
    """
    Generate perlin noise aggregate.

    Parameters
    ----------
    agg : xr.DataArray
        2D array of size width x height, will be used to determine
        height/ width and which platform to use for calculation.
    freq : tuple, default=(1,1)
        (x, y) frequency multipliers.
    seed : int, default=5
        Seed for random number generator.

    Returns
    -------
    perlin_agg : xarray.DataArray
        2D array of perlin noise values.

    References
    ----------
        - Paul Panzer: https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy # noqa
        - ICA: http://www.mountaincartography.org/mt_hood/pdfs/nighbert_bump1.pdf # noqa

    Examples
    --------
    .. plot::
       :include-source:

        import matplotlib.pyplot as plt
        import xarray as xr
        from xrspatial import fast_perlin
        noise = xr.DataArray(np.zeros((H, W), dtype=np.float32),
                        name='numpy_terrain',
                        dims=('y', 'x'),
                        attrs={'res': 1})

        # Generate Perlin Noise Aggregate
        perlin_default = fast_perlin(noise)

        # With Increased x Frequency
        perlin_high_x_freq = fast_perlin(noise, freq = (5, 1))

        # With Increased y Frequency
        perlin_high_y_freq = fast_perlin(noise, freq = (1, 5))

        # With a Different Seed
        perlin_seed_1 = fast_perlin(noise, seed = 1)

        # Plot Default Perlin
        perlin_default.plot(cmap = 'inferno', aspect = 2, size = 4)
        plt.title("Default")

        # Plot High x Frequency
        perlin_high_x_freq.plot(cmap = 'inferno', aspect = 2, size = 4)
        plt.title("High x Frequency")

        # Plot High y Frequency
        perlin_high_y_freq.plot(cmap = 'inferno', aspect = 2, size = 4)
        plt.title("High y Frequency")

        # Plot Seed = 1
        perlin_seed_1.plot(cmap = 'inferno', aspect = 2, size = 4)
        plt.title("Seed = 1")

    .. sourcecode:: python

        >>> print(perlin_default[200:203, 200: 202])
        <xarray.DataArray (y: 3, x: 2)>
        array([[0.56800979, 0.56477393],
               [0.56651744, 0.56331014],
               [0.56499184, 0.56181344]])
        Dimensions without coordinates: y, x
        Attributes:
            res:      1

        >>> print(perlin_high_x_freq[200:203, 200: 202])
        <xarray.DataArray (y: 3, x: 2)>
        array([[0.5       , 0.48999444],
               [0.5       , 0.48999434],
               [0.5       , 0.48999425]])
        Dimensions without coordinates: y, x
        Attributes:
            res:      1

        >>> print(perlin_high_y_freq[200:203, 200: 202])
        <xarray.DataArray (y: 3, x: 2)>
        array([[0.31872961, 0.31756859],
               [0.2999256 , 0.2988189 ],
               [0.28085118, 0.27979834]])
        Dimensions without coordinates: y, x
        Attributes:
            res:      1

        >>> print(perlin_seed_1[200:203, 200: 202])
        <xarray.DataArray (y: 3, x: 2)>
        array([[0.12991498, 0.12984185],
               [0.13451158, 0.13441514],
               [0.13916956, 0.1390495 ]])
        Dimensions without coordinates: y, x
        Attributes:
            res:      1
    """

    # numpy case
    if isinstance(agg.data, np.ndarray):
        out = _perlin_numpy(agg.data, freq, seed)
    # cupy case
    elif has_cuda() and isinstance(agg.data, cupy.ndarray):
        out = _perlin_cupy(agg.data, freq, seed)
    # dask + numpy case
    elif isinstance(agg.data, da.Array):
        out = _perlin_dask_numpy(agg.data, freq, seed)
    else:
        raise TypeError('Unsupported Array Type: {}'.format(type(agg.data)))

    result = xr.DataArray(out,
                          dims=agg.dims,
                          attrs=agg.attrs,
                          name=name)
    return result

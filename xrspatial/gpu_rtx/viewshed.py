# Only call functions in this file if has_rtx() returns True as this checks
# that the required dependent libraries are installed.

import cupy
import math
import numba as nb
import numpy as np
from rtxpy import RTX
from typing import Union
import xarray as xr

from .cuda_utils import add, diff, dot, float3, invert, make_float3, mul
from .mesh_utils import create_triangulation

from ..utils import calc_cuda_dims


# If a cell is invisible, its value is set to -1
INVISIBLE = -1


@nb.cuda.jit
def _generate_primary_rays_kernel(data, x_coords, y_coords, H, W):
    """
    A GPU kernel that given a set of x and y discrete coordinates on a raster
    terrain generates in @data a list of parallel rays that represent camera
    rays generated from an orthographic camera that is looking straight down
    at the surface from an origin height 10000.
    """
    i, j = nb.cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        if (j == W-1):
            data[i, j, 0] = x_coords[j] - 1e-3
        else:
            data[i, j, 0] = x_coords[j] + 1e-3

        if (i == H-1):
            data[i, j, 1] = y_coords[i] - 1e-3
        else:
            data[i, j, 1] = y_coords[i] + 1e-3

        data[i, j, 2] = 10000  # Location of the camera (height)
        data[i, j, 3] = 1e-3
        data[i, j, 4] = 0
        data[i, j, 5] = 0
        data[i, j, 6] = -1
        data[i, j, 7] = np.inf


def _generate_primary_rays(rays, x_coords, y_coords, H, W):
    griddim, blockdim = calc_cuda_dims((H, W))
    d_y_coords = cupy.array(y_coords)
    d_x_coords = cupy.array(x_coords)
    _generate_primary_rays_kernel[griddim, blockdim](
        rays, d_x_coords, d_y_coords, H, W)
    return 0


@nb.cuda.jit
def _calc_viewshed_kernel(hits, visibility_grid, H, W):
    i, j = nb.cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        dist = hits[i, j, 0]
        # We traced the viewshed rays and now hits contains the intersection
        # data.  If dist > 0, then we were able to hit something along the
        # length of the ray which means that the pixel we targeted is not
        # directly visible from the view point.
        if dist >= 0:
            visibility_grid[i, j] = INVISIBLE


def _calc_viewshed(hits, visibility_grid, H, W):
    griddim, blockdim = calc_cuda_dims((H, W))
    _calc_viewshed_kernel[griddim, blockdim](hits, visibility_grid, H, W)
    return 0


@nb.cuda.jit
def _generate_viewshed_rays_kernel(
    camera_rays, hits, vsrays, visibility_grid, H, W, vp
):
    i, j = nb.cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        observer_elev = vp[2]
        target_elev = vp[3]
        dist = hits[i, j, 0]  # distance to surface from camera
        # normal vector at intersection with surface
        norm = make_float3(hits[i, j], 1)
        if norm[2] < 0:  # if back hit, face forward
            norm = invert(norm)
        camera_ray = camera_rays[i, j]
        # get the camera ray origin
        ray_origin = make_float3(camera_ray, 0)
        # get the camera ray direction
        ray_dir = make_float3(camera_ray, 4)
        # calculate intersection point
        hit_p = add(ray_origin, mul(ray_dir, dist))
        # generate new ray origin, and a little offset to avoid
        # self-intersection
        new_origin = add(hit_p, mul(norm, 1e-3))
        # move the new origin up by the selected by user target_elev factor
        new_origin = add(new_origin, float3(0, 0, target_elev))

        w = int(vp[0])
        h = int(vp[1])
        # get the camera ray that was cast for the location of the viewshed
        # origin
        viewshed_ray = camera_rays[h, w]
        # get the distance from the camera to the viewshed point
        dist = hits[h, w, 0]
        # get the origin on the camera of the ray towards VP point
        ray_origin = make_float3(viewshed_ray, 0)
        # get the direction from camera to VP point
        ray_dir = make_float3(viewshed_ray, 4)
        # calculate distance from camera to VP
        hit_p = add(ray_origin, mul(ray_dir, dist))
        # calculate the VP location on the surface and add the VP offset
        viewshed_point = add(hit_p, float3(0, 0, observer_elev))

        # calculate vector from SurfaceHit to VP
        new_dir = diff(viewshed_point, new_origin)
        # calculate distance from surface to VP
        length = math.sqrt(dot(new_dir, new_dir))
        # normalize the direction (vector v)
        new_dir = mul(new_dir, 1/length)

        # cosine of the angle between n and v
        cosine = dot(norm, new_dir)
        theta = math.acos(cosine)  # Cosine angle in radians
        theta = (180*theta)/math.pi  # Cosine angle in degrees

        # prepare a viewshed ray to cast to determine visibility
        vsray = vsrays[i, j]
        vsray[0] = new_origin[0]
        vsray[1] = new_origin[1]
        vsray[2] = new_origin[2]
        vsray[3] = 0
        vsray[4] = new_dir[0]
        vsray[5] = new_dir[1]
        vsray[6] = new_dir[2]
        vsray[7] = length

        visibility_grid[i, j] = theta


def _generate_viewshed_rays(rays, hits, vsrays, visibility_grid, H, W, vp):
    griddim, blockdim = calc_cuda_dims((H, W))
    _generate_viewshed_rays_kernel[griddim, blockdim](
        rays, hits, vsrays, visibility_grid, H, W, vp)
    return 0


def _viewshed_rt(
    raster: xr.DataArray,
    optix: RTX,
    x: Union[int, float],
    y: Union[int, float],
    observer_elev: float,
    target_elev: float,
) -> xr.DataArray:

    H, W = raster.shape

    y_coords = raster.indexes.get('y').values
    x_coords = raster.indexes.get('x').values

    # validate x arg
    if x < x_coords.min():
        raise ValueError("x argument outside of raster x_range")
    elif x > x_coords.max():
        raise ValueError("x argument outside of raster x_range")

    # validate y arg
    if y < y_coords.min():
        raise ValueError("y argument outside of raster y_range")
    elif y > y_coords.max():
        raise ValueError("y argument outside of raster y_range")

    selection = raster.sel(x=[x], y=[y], method='nearest')
    x = selection.x.values[0]
    y = selection.y.values[0]

    y_view = np.where(y_coords == y)[0][0]
    x_view = np.where(x_coords == x)[0][0]

    # Device buffers
    d_rays = cupy.empty((H, W, 8), np.float32)
    d_hits = cupy.empty((H, W, 4), np.float32)
    d_visgrid = cupy.empty((H, W), np.float32)
    d_vsrays = cupy.empty((H, W, 8), np.float32)

    _generate_primary_rays(d_rays, x_coords, y_coords, H, W)
    device = cupy.cuda.Device(0)
    device.synchronize()
    res = optix.trace(d_rays, d_hits, W*H)
    if res:
        raise RuntimeError(f"Failed trace 1, error code: {res}")

    _generate_viewshed_rays(d_rays, d_hits, d_vsrays, d_visgrid, H, W,
                            (x_view, y_view, observer_elev, target_elev))
    device.synchronize()
    res = optix.trace(d_vsrays, d_hits, W*H)
    if res:
        raise RuntimeError(f"Failed trace 2, error code: {res}")

    _calc_viewshed(d_hits, d_visgrid, H, W)

    if isinstance(raster.data, np.ndarray):
        visgrid = cupy.asnumpy(d_visgrid)
    else:
        visgrid = d_visgrid

    view = xr.DataArray(
        visgrid,
        name="viewshed",
        coords=raster.coords,
        dims=raster.dims,
        attrs=raster.attrs)

    return view


def viewshed_gpu(
    raster: xr.DataArray,
    x: Union[int, float],
    y: Union[int, float],
    observer_elev: float,
    target_elev: float,
) -> xr.DataArray:
    if not isinstance(raster.data, cupy.ndarray):
        raise TypeError("raster.data must be a cupy array")

    optix = RTX()
    create_triangulation(raster, optix)

    return _viewshed_rt(raster, optix, x, y, observer_elev, target_elev)

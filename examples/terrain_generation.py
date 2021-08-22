import numpy as np
import datashader as ds
from datashader.transfer_functions import shade
from datashader.transfer_functions import stack
from datashader.colors import Elevation
import xarray as xr


from xrspatial import generate_terrain
# from xrspatial import hillshade
# from xrspatial import bump
# from xrspatial import mean
from xrspatial import perlin
import time
import argparse


from functools import partial

parser = argparse.ArgumentParser(description='Simple terrain generation example, to be used a benchmarking template.',
                                 usage='python terrain_generation.py -width width -height height -i iteration')

parser.add_argument('-width', '--width', type=int, default=600,
                    help='Canvas width.')
parser.add_argument('-height', '--height', type=int, default=400,
                    help='Canvas height.')

parser.add_argument('-i', '--iterations', type=int, default=1,
                    help='Number of times to repeat the function call.')

parser.add_argument('-g', '--gpu', action='store_true',
                    help='Use the GPU.')

if __name__ == '__main__':
    args = parser.parse_args()


    W = args.width
    H = args.height


    cvs = ds.Canvas(plot_width=W, plot_height=H,
                    # x_range=(-1, 1),
                    # y_range=(-1, 1))
                    x_range=(-20e6, 20e6),
                    y_range=(-20e6, 20e6))

    terrain = generate_terrain(canvas=cvs)

    if args.gpu is True:
        import cupy
        gpu_terrain = xr.DataArray(cupy.array(terrain.data), 
            name=terrain.name, 
            coords=terrain.coords,
            dims=terrain.dims,
            attrs=terrain.attrs)
        terrain=gpu_terrain
        #terrain.data = xr.DataArray(cupy.array(terrain.data))

    # src = terrain.data
    freq = (1, 1)
    seed = 0
    start = time.time()
    per = perlin(terrain, freq, seed)
    end = time.time()
    print('Warm-up time: {:.3f} sec.'.format(end-start))

    elapsed_sec = 0
    for i in range(args.iterations):
        start = time.time()
        per = perlin(terrain, freq, seed)
        elapsed_sec += time.time() - start

    print('Total time: {:.3f} sec.'.format(elapsed_sec))
    print('Time per call: {:.3f} sec.'.format(elapsed_sec/args.iterations))
    print('Result: ', per)


    # def heights(locations, src, src_range, height=20):
    #     num_bumps = locations.shape[0]
    #     out = np.zeros(num_bumps, dtype=np.uint16)
    #     for r in range(0, num_bumps):
    #         loc = locations[r]
    #         x = loc[0]
    #         y = loc[1]
    #         val = src[y, x]
    #         if val >= src_range[0] and val < src_range[1]:
    #             out[r] = height
    #     return out


    # T = 300000  # Number of trees to add per call
    # src = terrain.data
    # trees = bump(W, H, count=T, height_func=partial(heights, src=src,
    #              src_range=(1000, 1300), height=5))
    # trees += bump(W, H, count=T//2, height_func=partial(
    #         heights, src=src, src_range=(1300, 1700), height=20))
    # trees += bump(W, H, count=T//3, height_func=partial(
    #         heights, src=src, src_range=(1700, 2000), height=5))

    # tree_colorize = trees.copy()
    # tree_colorize.data[tree_colorize.data == 0] = np.nan

    # LAND_CONSTANT = 50.0

    # water = terrain.copy()
    # water.data = np.where(water.data > 0, LAND_CONSTANT, 0)
    # water = mean(water, passes=50, excludes=[LAND_CONSTANT])
    # water.data[water.data == LAND_CONSTANT] = np.nan


    # def create_map(azimuth):

    #     global cvs
    #     global terrain
    #     global water
    #     global trees

    #     img = stack(shade(terrain, cmap=Elevation, how='linear'),
    #                 shade(water, cmap=['aqua', 'white']),
    #                 shade(hillshade(terrain + trees, azimuth=azimuth),
    #                       cmap=['black', 'white'], how='linear', alpha=128),
    #                 shade(tree_colorize, cmap='limegreen', how='linear')
    #                 )

    #     print('image created')

    #     return img.to_pil()


    # def create_map2():

    #     global cvs
    #     global terrain
    #     global water
    #     global trees

    #     img = stack(shade(terrain, cmap=['black', 'white'], how='linear'))

    #     yield img.to_pil()

    #     img = stack(shade(terrain, cmap=Elevation, how='linear'))

    #     yield img.to_pil()

    #     img = stack(shade(terrain, cmap=Elevation, how='linear'),
    #                 shade(hillshade(terrain, azimuth=210),
    #                       cmap=['black', 'white'], how='linear', alpha=128),
    #                 )

    #     yield img.to_pil()

    #     img = stack(shade(terrain, cmap=Elevation, how='linear'),
    #                 shade(water, cmap=['aqua', 'white']),
    #                 shade(hillshade(terrain, azimuth=210),
    #                       cmap=['black', 'white'], how='linear', alpha=128),
    #                 )

    #     yield img.to_pil()

    #     img = stack(shade(terrain, cmap=Elevation, how='linear'),
    #                 shade(water, cmap=['aqua', 'white']),
    #                 shade(hillshade(terrain + trees, azimuth=210),
    #                       cmap=['black', 'white'], how='linear', alpha=128),
    #                 shade(tree_colorize, cmap='limegreen', how='linear')
    #                 )

    #     yield img.to_pil()
    #     yield img.to_pil()
    #     yield img.to_pil()
    #     yield img.to_pil()


    # def gif1():

    #     images = []

    #     for i in np.linspace(0, 360, 6):
    #         images.append(create_map(int(i)))

    #     images[0].save('animated_hillshade.gif',
    #                    save_all=True, append_images=images[1:],
    #                    optimize=False, duration=5000, loop=0)


    # def gif2():

    #     images = list(create_map2())

    #     images[0].save('composite_map.gif',
    #                    save_all=True, append_images=images[1:],
    #                    optimize=False, duration=1000, loop=0)


    # gif2()

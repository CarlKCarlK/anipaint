import logging
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from PIL.ImageDraw import Draw
from pysnptools.util.mapreduce1.runner import LocalMultiThread
from pysnptools.util.mapreduce1 import map_reduce


def composite(
    base_image, sprite, x, y, angle_degrees, sprite_factor, draw_debug_line=False
):
    # angle_degrees = math.degrees(angle_radians)-90
    # https://stackoverflow.com/questions/37941648/unable-to-crop-away-transparency-neither-pil-getbbox-nor-numpy-are-working
    angle_degrees = angle_degrees % 360.0

    if sprite_factor != 1:
        sprite = sprite.resize(
            (
                int(sprite.width * sprite_factor + 0.5),
                int(sprite.height * sprite_factor + 0.5),
            ),
            resample=Image.LANCZOS,
        )

    sprite = sprite.crop(sprite.convert("RGBA").getbbox())
    result = base_image.copy()

    x0 = x + math.cos(math.radians(angle_degrees)) * sprite.width * -0.5
    y0 = y + math.sin(math.radians(angle_degrees)) * sprite.width * -0.5

    x2 = x + math.cos(math.radians(angle_degrees)) * sprite.width * 0.5
    y2 = y + math.sin(math.radians(angle_degrees)) * sprite.width * 0.5
    if draw_debug_line:
        draw = Draw(result)
        draw.line([(x0, y0), (x2, y2)], fill="red", width=sprite.height)

    rot = sprite.rotate(-angle_degrees, expand=True)
    rot = rot.crop(rot.getbbox())
    x1 = int(x - rot.width / 2)
    y1 = int(y - rot.height / 2)
    result.paste(rot, (x1, y1), mask=rot)

    return result


def find_edge_distance(matte_path, max_distance=255, threshold=127):
    matte_image = Image.open(matte_path)
    assert matte_image.mode == "RGB", f"Expect images to be RGB, not {matte_image.mode}"
    rgb_array = np.array(matte_image)
    gray_array = np.zeros(
        (rgb_array.shape[0] + 2, rgb_array.shape[1] + 2), dtype="uint8"
    )
    gray_array[1:-1, 1:-1] = np.where(
        rgb_array.mean(axis=2) < threshold, 0, 254
    )  # 254 to give room for +1
    # plt.imshow(gray_array)
    # plt.show()
    # print("!!!cmk")
    expanded = grid_nD(gray_array)
    middle = expanded[:, :, 1, 1]
    logging.info("Finding distances")
    for i in range(1, min(max_distance + 1, 255)):
        minplus1 = np.min(expanded, axis=(-2, -1)) + 1
        next = np.minimum(middle, minplus1)
        expanded[:, :, 1, 1] = next
        logging.info(f"distance {i}")
        if middle.max() < 254:
            break
    # plt.imshow(np.where(middle == 0, 256, middle), cmap="twilight", vmin=0, vmax=255)
    # plt.show()
    # print("!!!cmk")
    return middle


def find_directions(middle, sigma=7):
    g0, g1 = np.gradient(middle)
    g0b = gaussian_filter(g0, sigma=sigma)
    g1b = gaussian_filter(g1, sigma=sigma)

    # plt.imshow(g0b, cmap="twilight")
    # plt.show()
    # plt.imshow(g1b, cmap="twilight")
    # plt.show()

    return g0b, g1b


def pre_cache_edge_distance(
    pattern, cache_folder, runner=None, max_distance=255, threshold=127
):
    pattern = Path(pattern)

    def mapper(matte_path):
        cached_edge_distance(
            matte_path, cache_folder, max_distance=max_distance, threshold=threshold
        )

    map_reduce(list(pattern.parent.glob(pattern.name)), mapper=mapper, runner=runner)


def cached_edge_distance(matte_path, cache_folder, max_distance=255, threshold=127):
    if ".edge_distance" in matte_path.name:
        return

    cache_folder = Path(cache_folder)

    cache_path = (cache_folder / matte_path.name).with_suffix(
        ".edge_distance{0}{1}.png".format(
            "" if max_distance == 255 else f".md{max_distance}",
            "" if threshold == 127 else f".th{threshold}",
        )
    )

    if cache_path.exists():
        cache_image = Image.open(cache_path)
        assert cache_image.mode == "L", f"Expect images to be L, not {cache_image.mode}"
        cache_array = np.array(cache_image)
        return cache_array

    matte_array = find_edge_distance(
        matte_path, max_distance=max_distance, threshold=threshold
    )

    matte_image = Image.fromarray(matte_array, mode="L")
    os.makedirs(cache_path.parent, exist_ok=True)
    matte_image.save(cache_path, optimize=True, compress_level=0)
    return matte_array


# See http://drsfenner.org/blog/2015/08/game-of-life-in-numpy-2/
def grid_nD(arr):
    assert all(_len > 2 for _len in arr.shape)

    nDims = len(arr.shape)
    newShape = [_len - 2 for _len in arr.shape]
    newShape.extend([3] * nDims)

    newStrides = arr.strides + arr.strides
    return as_strided(arr, shape=newShape, strides=newStrides)


# how_far_in could be a little random and based on width of stroke
def paint_edge(
    edge_distance,
    brush_image,
    how_far_in,
    credit_range,
    random_count,
    keep_threshold,
    seed=231,
    show_work=False,
):
    assert (
        brush_image.mode == "RGBA"
    ), f"Expect images to be RGBA, not {brush_image.mode}"

    candidates = np.nonzero(edge_distance == how_far_in)

    average_brush = int(np.array(brush_image)[:, :, 0:2].mean() + 0.5)
    count_these = (edge_distance >= credit_range[0]) * (edge_distance < credit_range[1])

    rng = np.random.RandomState(seed=seed)  # random number generator

    im1 = Image.new("RGBA", list(edge_distance.shape)[::-1], (0, 0, 0, 0))

    def how_dark(image):
        score = np.where(
            count_these, np.array(image)[:, :, 0:-1].mean(axis=-1), average_brush
        ).mean()
        return score

    old_darkness = how_dark(im1)
    for _ in range(random_count):
        i = rng.choice(len(candidates[1]))
        x, y = candidates[0][i], candidates[1][i]
        dx, dy = directions[0][x, y], directions[1][x, y]
        im2 = composite(
            im1,
            brush_image,
            y,
            x,
            -math.degrees(math.atan2(dy, dx)),
            sprite_factor=1,
            draw_debug_line=False,
        )
        new_darkness = how_dark(im2)
        diff = new_darkness - old_darkness
        if show_work:
            print(f"{diff:,}")
            plt.plot(y, x, "o")
            # plt.quiver(y,x,-dyr,-dxr,angles='xy',width=.002)
            plt.imshow(im2)
            plt.show()
        if diff > keep_threshold:
            old_darkness = new_darkness
            im1 = im2
        elif show_work:
            logging.info("don't keep")
    # plt.imshow(im1)
    # im1.save(tmp_path / f"edge{how_far_in}_{random_count}_{keep_threshold}.png")
    return im1
    # plt.imshow(im1)
    # plt.show()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    shared_datadir = Path(r"D:\OneDrive\programs\anipaint\anipaint\tests\data")
    tmp_path = Path(r"m:/deldir/anipaint/tests")

    # open a matte
    matte_file = "Comp 2/ShirtMAtte_00000.jpg"
    matte_path = shared_datadir / matte_file

    edge_distance = cached_edge_distance(matte_path, shared_datadir / "Comp 2")

    # edge_distance = find_edge_distance(matte_path, max_distance=10)
    directions = find_directions(edge_distance)

    runner = LocalMultiThread(12)

    pre_cache_edge_distance(
        r"E:\Dropbox\Watercolor Animation Assets\Comp 2\Comp 2\*.*",
        r"E:\Dropbox\Watercolor Animation Assets\Comp 2\Comp 2\cache",
        runner=runner,
    )

    brush_file = "brushes/PaintStrokes (0-00-00-04).png"
    brush_image = Image.open(shared_datadir / brush_file)

    im1 = paint_edge(
        edge_distance,
        brush_image,
        how_far_in=10,
        credit_range=[1, 20],
        random_count=100,
        keep_threshold=0.1,
    )
    plt.imshow(im1)
    plt.show()

    print("!!!cmk")

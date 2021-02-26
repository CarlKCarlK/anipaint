import logging
import math
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
from PIL.ImageDraw import Draw
from pysnptools.util.mapreduce1 import map_reduce
from scipy.ndimage.filters import gaussian_filter


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
    logging.info(f"Finding distances for '{matte_path.name}'")
    for i in range(1, min(max_distance + 1, 255)):
        minplus1 = np.min(expanded, axis=(-2, -1)) + 1
        next = np.minimum(middle, minplus1)
        expanded[:, :, 1, 1] = next
        # logging.info(f"distance {i}")
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


def cached_edge_distance(
    matte_path, cache_folder=None, max_distance=255, threshold=127
):
    if ".edge_distance" in matte_path.name:
        return

    if cache_folder is not None:
        cache_folder = Path(cache_folder)
    else:
        cache_folder = matte_path.parent / "cache"

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


def paint(
    output_folder,  # set to None to return frames instead
    matte_pattern,
    brush_pattern,
    random_count,
    candidate_range=(1, 256),
    credit_range=(1, 256),
    mixing_range=(0, 1),
    outside_penalty=0,
    keep_threshold=0.0,
    paint_same_threshold=0.0,
    default_angle_degrees=15,
    default_angle_sd=5,
    sprite_factor_range=(1.0, 1.0),  # both inclusive
    runner=None,
    cache_folder=None,
    seed=231,
    show_work=False,
):
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    matte_path_list = sorted(matte_pattern.parent.glob(matte_pattern.name))

    # !!!cmk could pass list, could only open each file once in find_same
    skip_list = [diff <= paint_same_threshold for diff in find_same(matte_pattern)]

    def mapper(matte_path_and_skip):
        matte_path, skip = matte_path_and_skip
        print(f"painting '{matte_path.name}'")  # cmk should log

        if output_folder is not None:
            output_path = (output_folder / matte_path.name).with_suffix(".output.png")
            if output_path.exists():
                print(
                    f"Output already exists, so skipping ('{output_path.name}'')"
                )  # cmk should log as warn????
                return output_path

        if skip:
            return None

        image = paint_one(
            matte_path,
            brush_pattern,
            random_count=random_count,
            outside_penalty=outside_penalty,
            keep_threshold=keep_threshold,
            candidate_range=candidate_range,
            credit_range=credit_range,
            mixing_range=mixing_range,
            sprite_factor_range=sprite_factor_range,
            default_angle_degrees=default_angle_degrees,
            default_angle_sd=default_angle_sd,
            cache_folder=cache_folder,
            seed=seed,
            show_work=show_work,
        )
        if output_folder is None:
            return image
        else:
            image.save(output_path, optimize=True, compress_level=0)
            return output_path

    result_w_skip_list = map_reduce(
        zip(matte_path_list, skip_list), mapper=mapper, runner=runner
    )

    result_list = []
    previous_noskip_result = None
    for result_w_skip, matte_path in zip(result_w_skip_list, matte_path_list):
        if result_w_skip is None:
            if output_folder is None:
                result_list.append(previous_noskip_result)
            else:
                output_path = (output_folder / matte_path.name).with_suffix(
                    ".output.png"
                )  # !!!cmk similar code elsewhere
                shutil.copy(previous_noskip_result, output_path)
                result_list.append(output_path)
        else:
            previous_noskip_result = result_w_skip
            result_list.append(previous_noskip_result)

    return result_list


def paint_one(
    matte_path,
    brush_pattern,
    random_count,
    candidate_range=(1, 256),
    credit_range=(1, 256),
    mixing_range=(0, 1),
    outside_penalty=0,
    keep_threshold=0.5,
    default_angle_degrees=15,
    default_angle_sd=5,
    sprite_factor_range=(1.0, 1.0),  # both inclusive
    cache_folder=None,
    seed=231,
    show_work=False,
):
    edge_distance = cached_edge_distance(matte_path, cache_folder=cache_folder)
    assert (
        candidate_range[0] < candidate_range[1]
    ), "first value in candidate_range must be less than the 2nd"
    assert (
        credit_range[0] < credit_range[1]
    ), "first value in credit_range must be less than the 2nd"
    assert (
        mixing_range[0] < mixing_range[1]
    ), "first value in start_mixing_range must be less than the 2nd"
    assert (
        0 < sprite_factor_range[0] <= sprite_factor_range[1]
    ), "first value of sprite_factor_range must be more than 0 and less than or equal to the 2nd"

    brush_pattern = Path(brush_pattern)
    brush_list = []
    for brush_path in brush_pattern.parent.glob(brush_pattern.name):
        brush_image = Image.open(brush_path)
        assert (
            brush_image.mode == "RGBA"
        ), f"Expect brush_image to be RGBA, not {brush_image.mode}"
        brush_list.append(brush_image)

    pre_candidate_points = (edge_distance >= candidate_range[0]) * (
        edge_distance < candidate_range[1]
    )

    credit_points = (edge_distance >= credit_range[0]) * (
        edge_distance < credit_range[1]
    )

    penalty_points = edge_distance == 0

    directions = find_directions(edge_distance)
    rng = np.random.RandomState(seed=seed)  # random number generator

    current_image = Image.new("RGBA", list(edge_distance.shape)[::-1], (0, 0, 0, 0))

    def how_dark(image):
        darkness_array = np.array(image)[:, :, 0:-1].sum(
            axis=-1
        )  # does darkness, not mask
        score = np.where(credit_points, darkness_array, 0).sum()
        if outside_penalty != 0:
            score -= np.where(penalty_points, darkness_array, 0).sum() * outside_penalty

        return score

    old_score = 0
    for _ in range(random_count):

        # !!!cmk np.array slow?
        candidate_points = np.nonzero(
            np.where(pre_candidate_points, np.array(current_image)[:, :, 3] == 0, 0)
        )

        candidates_len = len(candidate_points[0])
        if candidates_len == 0:
            break
        i = rng.choice(candidates_len)
        x, y = candidate_points[0][i], candidate_points[1][i]
        v = edge_distance[x, y]
        fraction_interior = np.clip(
            (v - mixing_range[0]) / (mixing_range[1] - mixing_range[0]), 0, 1,
        )
        dxe, dye = directions[0][x, y], directions[1][x, y]
        random_angle_degrees = rng.normal(default_angle_degrees, default_angle_sd)
        dxi, dyi = (
            math.cos(math.radians(random_angle_degrees)),
            math.sin(math.radians(random_angle_degrees)),
        )
        dx, dy = (
            fraction_interior * dxi + (1 - fraction_interior) * dxe,
            fraction_interior * dyi + (1 - fraction_interior) * dye,
        )
        angle_degrees = math.degrees(math.atan2(dy, dx))

        brush_index = rng.choice(len(brush_list))
        brush_image = brush_list[brush_index]

        sprite_factor = (
            math.exp(
                rng.uniform(
                    math.log(sprite_factor_range[0] ** 2),
                    math.log(sprite_factor_range[1] ** 2),
                )
            )
            ** 0.5
            if sprite_factor_range[0] < sprite_factor_range[1]
            else sprite_factor_range[0]
        )
        if sprite_factor != 1:
            brush_image = brush_image.resize(
                (
                    int(brush_image.width * sprite_factor + 0.5),
                    int(brush_image.height * sprite_factor + 0.5),
                ),
                resample=Image.LANCZOS,
            )
        best_improvement = np.array(brush_image)[:, :, 0:-1].sum()

        possible_image = composite(
            current_image,
            brush_image,
            y,
            x,
            -angle_degrees,
            sprite_factor=1,
            draw_debug_line=False,
        )

        new_score = how_dark(possible_image)
        fraction_new = (int(new_score) - int(old_score)) / best_improvement
        if show_work:
            print(f"{fraction_new:,}")
            plt.plot(y, x, "o")
            # plt.quiver(y,x,-dy,-dx,angles='xy',width=.002)
            plt.imshow(possible_image)
            plt.show()
        if fraction_new > keep_threshold:
            old_score = new_score
            current_image = possible_image
        elif show_work:
            print("don't keep")
    # plt.imshow(im_in)
    # im_in.save(tmp_path / f"inner{start_distance}_{random_count}_{keep_threshold}.png")
    # plt.imshow(im_in)
    # plt.show()
    return current_image


def pairs(sequence):
    before = None
    for item in sequence:
        yield before, item
        before = item


def find_same(matte_pattern):
    for before_path, after_path in pairs(
        sorted(matte_pattern.parent.glob(matte_pattern.name))
    ):
        if before_path is None:
            yield 256.0
            continue
        before_array = np.array(Image.open(before_path))
        after_array = np.array(Image.open(after_path))
        result = np.abs(before_array - after_array).mean()
        # print(before_path.name, after_path.name, result)
        yield result


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    folder = Path(r"m:\deldir\Watercolor Animation Assets")

    # result = list(find_same(matte_pattern=folder / "Comp 2/Comp 2/*.jpg"))

    folder = Path(r"m:\deldir\Watercolor Animation Assets")
    brush_pattern = folder / "brushes/*.png"

    paint(
        output_folder=folder / "SkinMatte/Comp 2/outputs/run6a_4",
        matte_pattern=folder / "SkinMatte/Comp 2/Comp 2_0000*.jpg",
        brush_pattern=folder / "brushes/*.png",
        random_count=5,
        outside_penalty=4,
        keep_threshold=0,
        candidate_range=(1, 256),
        credit_range=(1, 256),
        mixing_range=(255, 256),
        sprite_factor_range=(0.25, 1),
        paint_same_threshold=5.0,
        runner=None,
    )

    print("!!!cmk")

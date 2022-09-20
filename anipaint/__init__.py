# !!!cmk Easy: Stop on failure & remember all failures and report all at end
# !!!cmk Some way to do multi-levels of input, not ignoring output, and have multi output levels as wanted.


import glob
import json
import logging
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image, ImageFilter, ImageOps
from PIL.ImageDraw import Draw
from pysnptools.util.mapreduce1 import map_reduce
from scipy.ndimage import gaussian_filter


def composite(
    base_image, sprite, x, y, angle_degrees, sprite_factor=1, draw_debug_line=False
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
    if x1 >= 0 and y1 >= 0:
        result.alpha_composite(rot, dest=(x1, y1))
    else:
        new_stamp = Image.new("RGBA", result.size, (0, 0, 0, 0))
        new_stamp.paste(rot, (x1, y1))
        result.alpha_composite(new_stamp)

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
    expanded = grid_nD(gray_array)
    middle = expanded[:, :, 1, 1]
    logging.info(f"Finding distances for '{matte_path.name}'")
    for i in range(1, min(max_distance + 1, 255)):
        min_plus1 = np.min(expanded, axis=(-2, -1)) + 1
        next = np.minimum(middle, min_plus1)
        expanded[:, :, 1, 1] = next
        # logging.info(f"distance {i}")
        if middle.max() < 254:
            break
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


# cmk def pre_cache_edge_distance(
#     pattern, cache_folder, runner=None, max_distance=255, threshold=127
# ):
#     pattern = Path(pattern)

#     def mapper(matte_path):
#         cached_edge_distance(
#             matte_path, cache_folder, max_distance=max_distance, threshold=threshold
#         )

#     map_reduce(list(pattern.parent.glob(pattern.name)), mapper=mapper, runner=runner)


# See http://drsfenner.org/blog/2015/08/game-of-life-in-numpy-2/
def grid_nD(arr):
    assert all(_len > 2 for _len in arr.shape)

    nDims = len(arr.shape)
    newShape = [_len - 2 for _len in arr.shape]
    newShape.extend([3] * nDims)

    newStrides = arr.strides + arr.strides
    return as_strided(arr, shape=newShape, strides=newStrides)


@dataclass
class Paint:
    output_folder: Any
    matte_pattern: Any
    brush_pattern: Any
    stroke_count_max: int
    preview_frame: int = None
    batch_count: int = 1
    background_pattern: Any = None
    background_matte_blur: float = None
    candidate_range: Tuple[int] = (1, 256)
    credit_range: Tuple[int] = (1, 256)
    mixing_range: Tuple[int] = (0, 1)
    penalty_area_pixels_max: float = None
    brush_efficiency_min: float = None
    frames_diff_fraction_max: float = None
    default_angle_degrees: float = 15
    default_angle_sd: float = 5
    sprite_factor_range: Tuple[float] = (1.0, 1.0)  # both inclusive
    frame_runner: Any = None
    preview_runner: Any = None
    cache_folder: Any = None
    seed: int = 231

    def __post_init__(self):
        assert (
            self.candidate_range[0] < self.candidate_range[1]
        ), "first value in candidate_range must be less than the 2nd"
        assert (
            self.credit_range[0] < self.credit_range[1]
        ), "first value in credit_range must be less than the 2nd"
        assert (
            self.mixing_range[0] < self.mixing_range[1]
        ), "first value in start_mixing_range must be less than the 2nd"
        assert (
            0 < self.sprite_factor_range[0] <= self.sprite_factor_range[1]
        ), "first value of sprite_factor_range must be more than 0 and less than or equal to the 2nd"

        self.matte_pattern = Path(self.matte_pattern)
        if self.cache_folder is not None:
            self.cache_folder = Path(self.cache_folder)
        else:
            self.cache_folder = self.matte_pattern.parent / "cache"
        os.makedirs(self.cache_folder, exist_ok=True)

        if self.preview_frame is None:
            os.makedirs(self.output_folder, exist_ok=True)
        self.matte_path_list = sorted(
            self.matte_pattern.parent.glob(self.matte_pattern.name)
        )
        self.skip_list = self._find_skips(self.matte_path_list)

        if self.preview_frame is not None:
            self.matte_path_list = [self.matte_path_list[self.preview_frame]]
            self.skip_list = [False]
            self.frame_runner = None
        else:
            self.preview_runner = None

        self.brush_list = self.load_images(self.brush_pattern)
        self.background_list = (
            self.load_images(self.background_pattern)
            if self.background_pattern is not None
            else None
        )

    @staticmethod
    def save(preset, **kwargs):
        preset = Path(preset)
        assert not preset.exists(), "Preset already exists."
        after = {}
        for key, value in kwargs.items():
            if isinstance(value, Path):
                value = str(value)
            after[key] = value
        with open(preset.with_suffix(".temp.json"), "w") as f:
            json.dump(after, f)
        shutil.move(preset.with_suffix(".temp.json"), preset)

    @staticmethod
    def last_digits_to_star(matte_file):
        star = re.sub(r"(.*\D)(\d+)(.[^.]+)", r"\g<1>*\g<3>", str(matte_file))
        return Path(star)

    # Priority
    #     The function inputs (highest)
    #     The name of the matte file
    #     The preset (lowest)
    @staticmethod
    def batch(**kwargs):
        if "preset_folder" in kwargs:
            preset_folder = Path(kwargs["preset_folder"])
            del kwargs["preset_folder"]
        else:
            preset_folder = None

        assert "matte_pattern" in kwargs, "Expect 'matte_pattern' input"
        matte_pattern = Path(kwargs["matte_pattern"])
        matte_pattern_list = sorted(
            {
                Paint.last_digits_to_star(Path(matte_file))
                for matte_file in glob.glob(str(matte_pattern))
            }
        )

        for matte_pattern in matte_pattern_list:
            logging.info(f"Working on '{matte_pattern.name}'")
            try:

                # !!!!cmk continue even of something goes wrong

                name_dict = {}
                for name_piece in str(matte_pattern.name).split("_"):
                    if name_piece.endswith(")"):
                        key, val = name_piece[:-1].split("(")
                        name_dict[key] = val

                if "preset" in name_dict:
                    assert (
                        preset_folder is not None
                    ), "If a preset is given in matte name, expect a preset_folder"
                    preset_file = preset_folder / (name_dict["preset"] + ".preset.json")
                    assert (
                        preset_file.exists()
                    ), f"Expect preset file to exists '{str(preset_file)}'"

                    with open(preset_file) as f:
                        paint_dict = json.load(f)
                    del name_dict["preset"]
                else:
                    paint_dict = {}

                for key, value in name_dict.items():
                    paint_dict[key] = value

                for key, value in kwargs.items():
                    paint_dict[key] = value

                paint_dict["matte_pattern"] = matte_pattern

                Paint(**paint_dict).paint()
            except Exception as e:
                logging.warn(
                    f"Something went wrong; skipping to next matte pattern. ('{e}'')"
                )

    def load_images(self, pattern):
        result_list = []
        pattern = Path(pattern)
        for path in pattern.parent.glob(pattern.name):
            image = Image.open(path).copy()
            assert image.mode == "RGBA", f"Expect image to be RGBA, not {image.mode}"
            result_list.append(image)
        logging.info(f"Loaded {len(result_list)} image(s) from '{pattern}'")
        assert len(result_list) > 0, f"Expect at least one image from '{pattern}'"
        return result_list

    def paint(self):
        outer_count = -(-self.stroke_count_max // self.batch_count)  # round up

        def mapper(frame_index_and_matte_path_and_skip):
            frame_index, (matte_path, skip) = frame_index_and_matte_path_and_skip
            logging.info(f"painting '{matte_path.name}' (skip? {skip}) ")

            if self.preview_frame is None:
                output_path = self.create_output_path(matte_path)
                if output_path.exists():
                    logging.warning(
                        f"Output already exists, so skipping ('{output_path.name}'')"
                    )
                    return output_path

            if skip:
                return None

            image = self.paint_one(matte_path, outer_count, frame_index)
            if self.preview_frame is not None:
                return image
            else:
                image.save(output_path, optimize=True, compress_level=0)
                return output_path

        result_w_skip_list = map_reduce(
            list(enumerate(zip(self.matte_path_list, self.skip_list))),
            mapper=mapper,
            runner=self.frame_runner,
        )
        result_list = self.fill_skips(result_w_skip_list)
        if self.preview_frame is not None:
            result_list[0].show()

    def fill_skips(self, result_w_skip_list):
        result_list = []
        before_no_skip_result = None
        for result_w_skip, matte_path in zip(result_w_skip_list, self.matte_path_list):
            if result_w_skip is None:
                if self.preview_frame is not None:
                    result_list.append(before_no_skip_result)
                else:
                    output_path = self.create_output_path(matte_path)
                    shutil.copy(before_no_skip_result, output_path)
                    result_list.append(output_path)
            else:
                before_no_skip_result = result_w_skip
                result_list.append(before_no_skip_result)
        return result_list

    def create_output_path(self, matte_path):
        output_path = (self.output_folder / matte_path.name).with_suffix(".output.png")
        return output_path

    def cached_edge_distance(self, matte_path):
        max_distance = 255
        threshold = 127

        cache_path = (self.cache_folder / matte_path.name).with_suffix(
            ".edge_distance{0}{1}.png".format(
                "" if max_distance == 255 else f".md{max_distance}",
                "" if threshold == 127 else f".th{threshold}",
            )
        )

        if cache_path.exists():
            cache_image = Image.open(cache_path)
            assert (
                cache_image.mode == "L"
            ), f"Expect images to be L, not {cache_image.mode}"
            cache_array = np.array(cache_image)
            return cache_array

        matte_array = find_edge_distance(
            matte_path, max_distance=max_distance, threshold=threshold
        )

        matte_image = Image.fromarray(matte_array, mode="L")
        os.makedirs(cache_path.parent, exist_ok=True)
        matte_image.save(cache_path, optimize=True, compress_level=0)
        return matte_array

    def find_score(self, current_image, candidate, credit_area, penalty_area):
        if self.brush_efficiency_min is None and self.penalty_area_pixels_max is None:
            return 0, 0

        image = self.create_possible_image(current_image, candidate)
        image_opacity = np.array(image)[:, :, -1]  # opacity of every pixel 0..256
        # !!!cmk similar code elsewhere

        if self.brush_efficiency_min is not None:
            credit_area_pixels_covered = (
                np.where(credit_area, image_opacity, 0).sum() / 256.0
            )
        else:
            credit_area_pixels_covered = 0

        # !!!cmk similar code elsewhere
        if self.penalty_area_pixels_max is not None:
            penalty_area_pixels_covered = (
                np.where(penalty_area, image_opacity, 0).sum() / 256.0
            )
        else:
            penalty_area_pixels_covered = 0

        return credit_area_pixels_covered, penalty_area_pixels_covered

    def paint_one(self, matte_path, outer_count, frame_index):
        edge_distance = self.cached_edge_distance(matte_path)
        pre_candidate_points = (edge_distance >= self.candidate_range[0]) * (
            edge_distance < self.candidate_range[1]
        )
        credit_area = (edge_distance >= self.credit_range[0]) * (
            edge_distance < self.credit_range[1]
        )
        penalty_area = edge_distance == 0
        directions = find_directions(edge_distance)

        current_image = Image.new("RGBA", list(edge_distance.shape)[::-1], (0, 0, 0, 0))

        for outer_index in range(outer_count):

            image_opacity = np.array(current_image)[:, :, -1]

            candidate_points = np.nonzero(
                np.where(pre_candidate_points, image_opacity == 0, 0)
            )
            candidates_len = len(candidate_points[0])
            if candidates_len == 0:
                break

            if self.brush_efficiency_min is not None:
                old_credit_area_pixels_covered = (
                    np.where(credit_area, image_opacity, 0).sum() / 256.0
                )
            else:
                old_credit_area_pixels_covered = 0

            if self.penalty_area_pixels_max is not None:
                old_penalty_area_pixels_covered = (
                    np.where(penalty_area, image_opacity, 0).sum() / 256.0
                )
            else:
                old_penalty_area_pixels_covered = 0

            def mapper(batch_index):

                inner_seed = self.seed ^ (
                    batch_index
                    + self.batch_count * (outer_index + outer_count * frame_index)
                )
                # print(f"inner_seed {inner_seed}")
                candidate = self.random_candidate(
                    candidate_points,
                    edge_distance,
                    directions,
                    seed=inner_seed,
                )

                (
                    brush_efficiency,
                    new_penalty_area_pixels_covered,
                ) = self.find_brush_efficiency(
                    current_image,
                    candidate,
                    credit_area,
                    penalty_area,
                    old_credit_area_pixels_covered,
                )
                if (
                    self.brush_efficiency_min is None
                    or (brush_efficiency >= self.brush_efficiency_min)
                ) and (
                    self.penalty_area_pixels_max is None
                    or (
                        (
                            new_penalty_area_pixels_covered
                            - old_penalty_area_pixels_covered
                        )
                        <= self.penalty_area_pixels_max
                    )
                ):
                    return candidate
                else:
                    return None

            result_list = map_reduce(
                range(self.batch_count), mapper=mapper, runner=self.preview_runner
            )
            for candidate in result_list:
                if candidate is not None:
                    current_image = self.create_possible_image(current_image, candidate)

        if self.background_list is not None:
            matte_image = Image.open(matte_path)
            if self.background_matte_blur is not None:
                matte_image = matte_image.filter(
                    ImageFilter.GaussianBlur(self.background_matte_blur)
                )
            frame_seed = self.seed ^ frame_index
            # print(f"frame_seed {frame_seed}")
            rng = np.random.RandomState(seed=frame_seed)
            background = self.background_list[rng.choice(len(self.background_list))]
            background = self.tile_if_needed(background, matte_image)
            result = Image.new("RGBA", matte_image.size, (0, 0, 0, 0))
            result.paste(background, mask=ImageOps.grayscale(matte_image))
            result.alpha_composite(current_image)
            current_image = result
        return current_image

    @staticmethod
    def tile_if_needed(background, matte_image):
        if background.size != matte_image.size:
            new_background = Image.new("RGB", matte_image.size, (0, 0, 0))
            for x in range(0, matte_image.size[0], background.size[0]):
                for y in range(0, matte_image.size[1], background.size[1]):
                    new_background.paste(background, (x, y))
            background = new_background
        return background

    def find_brush_efficiency(
        self,
        current_image,
        candidate,
        credit_area,
        penalty_area,
        old_credit_area_pixels_covered,
    ):
        new_credit_area_pixels_covered, penalty_area_pixels_covered = self.find_score(
            current_image, candidate, credit_area, penalty_area
        )

        if self.brush_efficiency_min is not None:
            brush_pixels_covered = (
                np.array(candidate["brush_image"])[:, :, -1]
            ).sum() / 256.0
            brush_efficiency = (
                new_credit_area_pixels_covered - old_credit_area_pixels_covered
            ) / brush_pixels_covered
        else:
            brush_efficiency = 1.0
        return (
            brush_efficiency,
            penalty_area_pixels_covered,
        )

    def create_possible_image(self, current_image, candidate):
        possible_image = composite(
            current_image,
            candidate["brush_image"],
            candidate["y"],
            candidate["x"],
            -candidate["angle_degrees"],
        )
        return possible_image

    def random_candidate(self, candidate_points, edge_distance, directions, seed):
        # print(seed)
        rng = np.random.RandomState(seed=seed)
        candidates_len = len(candidate_points[0])
        i = rng.choice(candidates_len)
        # print(seed, candidates_len, i)
        x, y = candidate_points[0][i], candidate_points[1][i]
        angle_degrees = self.find_angle(x, y, edge_distance, directions, rng)
        brush_image = self.find_brush(rng)
        candidate = {
            "brush_image": brush_image,
            "x": x,
            "y": y,
            "angle_degrees": angle_degrees,
        }
        return candidate

    def find_brush(self, rng):
        brush_image = self.brush_list[rng.choice(len(self.brush_list))]

        sprite_factor = (
            math.exp(
                rng.uniform(
                    math.log(self.sprite_factor_range[0] ** 2),
                    math.log(self.sprite_factor_range[1] ** 2),
                )
            )
            ** 0.5
            if self.sprite_factor_range[0] < self.sprite_factor_range[1]
            else self.sprite_factor_range[0]
        )
        if sprite_factor != 1:
            brush_image = brush_image.resize(
                (
                    int(brush_image.width * sprite_factor + 0.5),
                    int(brush_image.height * sprite_factor + 0.5),
                ),
                resample=Image.LANCZOS,
            )
        return brush_image

    def find_angle(self, x, y, edge_distance, directions, rng):
        v = edge_distance[x, y]
        fraction_interior = np.clip(
            (v - self.mixing_range[0]) / (self.mixing_range[1] - self.mixing_range[0]),
            0,
            1,
        )
        dxe, dye = directions[0][x, y], directions[1][x, y]
        random_angle_degrees = rng.normal(
            self.default_angle_degrees, self.default_angle_sd
        )
        dxi, dyi = (
            math.cos(math.radians(random_angle_degrees)),
            math.sin(math.radians(random_angle_degrees)),
        )
        dx, dy = (
            fraction_interior * dxi + (1 - fraction_interior) * dxe,
            fraction_interior * dyi + (1 - fraction_interior) * dye,
        )
        angle_degrees = math.degrees(math.atan2(dy, dx))
        return angle_degrees

    def _find_skips(self, sorted_matte_path_list):

        if len(sorted_matte_path_list) == 0:
            logging.info("Empty cache_pattern")
            return []

        first = sorted_matte_path_list[0]
        last = sorted_matte_path_list[-1]
        cache_path = self.cache_folder / (
            Path(first.name).with_suffix(f".{last.stem}.diff_list.txt")
        )

        if cache_path.exists():
            logging.info(f"Loading '{cache_path}'")
            skip_list = []
            before_path = None
            with open(cache_path, "r") as f:
                for line, after_path in zip(f.readlines(), sorted_matte_path_list):
                    fields = line.strip().split("\t")
                    assert (
                        len(fields) == 3
                        and fields[1] == str(before_path)
                        and fields[2] == str(after_path)
                    ), f"diff_list file doesn't match cache_pattern ('{cache_path}'')"
                    diff = float(fields[0])
                    skip = (
                        self.frames_diff_fraction_max is not None
                        and diff < self.frames_diff_fraction_max
                    )
                    skip_list.append(skip)
                    before_path = after_path
            assert len(skip_list) == len(
                sorted_matte_path_list
            ), f"diff_list file doesn't match cache_pattern ('{cache_path}'')"
            return skip_list

        skip_list = []
        reference_array = None
        before_array = None
        before_path = None
        with open(cache_path.with_suffix(".txt.temp"), "w") as f:
            for after_path in sorted_matte_path_list:
                after_array = np.array(Image.open(after_path), "int16")
                if reference_array is None:
                    diff = 1.0
                else:
                    diff = np.abs(reference_array - after_array).mean() / 256.0
                    if reference_array is not before_array:
                        diff = max(
                            diff, np.abs(before_array - after_array).mean() / 256.0
                        )
                skip = (
                    self.frames_diff_fraction_max is not None
                    and diff < self.frames_diff_fraction_max
                )
                skip_list.append(skip)
                logging.info(
                    f"'{after_path.name}', diff from ref&before {diff:.7f}, skip? {skip}"
                )
                f.write(f"{diff}\t{str(before_path)}\t{str(after_path)}\n")
                before_array = after_array
                before_path = after_path
                if not skip:
                    reference_array = after_array
        shutil.move(cache_path.with_suffix(".txt.temp"), cache_path)

        return skip_list


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # folder = Path(r"m:\deldir\Watercolor Animation Assets")
    # result = list(find_same(matte_pattern=folder / "Comp 2/Comp 2/*.jpg"))

    folder = Path(r"m:\deldir\Watercolor Animation Assets")
    brush_pattern = folder / "brushes/*.png"

    Paint(
        output_folder=folder / "SkinMatte/Comp 2/outputs/run_test1",
        matte_pattern=folder / "SkinMatte/Comp 2/*.jpg",
        brush_pattern=folder / "brushes/*.png",
        stroke_count_max=5,
        penalty_area_pixels_max=4,
        brush_efficiency_min=0,
        candidate_range=(1, 256),
        credit_range=(1, 256),
        mixing_range=(255, 256),
        sprite_factor_range=(0.25, 1),
        frames_diff_fraction_max=0.02,  # fraction difference
        frame_runner=None,
    ).paint()

    print("!!!cmk")

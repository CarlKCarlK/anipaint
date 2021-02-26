import os
from pathlib import Path

from pysnptools.util.mapreduce1 import map_reduce


def runa(runner):
    from anipaint import paint

    folder = Path(r"E:\Dropbox\Watercolor Animation Assets")
    output_folder = folder / "SkinMatte/Comp 2/outputs/run6a"
    matte_pattern = folder / "SkinMatte/Comp 2/*.*"
    brush_pattern = folder / "brushes/*.png"

    os.makedirs(output_folder, exist_ok=True)

    def mapper(matte_path):
        print(matte_path)  # cmk should log
        output_path = (output_folder / matte_path.name).with_suffix(".output.png")
        if output_path.exists():
            return

        im_in = paint(
            matte_path,
            brush_pattern,
            random_count=500,
            outside_penalty=2,
            keep_threshold=0,
            candidate_range=(1, 256),
            credit_range=(1, 256),
            mixing_range=(255, 256),
            sprite_factor_range=(0.25, 1),
        )
        im_in.save(output_path, optimize=True, compress_level=0)

    map_reduce(
        list(matte_pattern.parent.glob(matte_pattern.name)),
        mapper=mapper,
        runner=runner,
    )


def runb(runner):
    from anipaint import paint

    folder = Path(r"E:\Dropbox\Watercolor Animation Assets")
    output_folder = folder / "Comp 2/Comp 2/outputs/run6b"
    matte_pattern = folder / "Comp 2/Comp 2/*.*"
    brush_pattern = folder / "brushes/*.png"

    os.makedirs(output_folder, exist_ok=True)

    def mapper(matte_path):
        print(matte_path)  # cmk should log
        output_path = (output_folder / matte_path.name).with_suffix(".output.png")
        if output_path.exists():
            return

        im_in = paint(
            matte_path,
            brush_pattern,
            random_count=500,
            outside_penalty=2,
            keep_threshold=0,
            candidate_range=(1, 256),
            credit_range=(1, 256),
            mixing_range=(255, 256),
            sprite_factor_range=(0.25, 1),
        )
        im_in.save(output_path, optimize=True, compress_level=0)

    map_reduce(
        list(matte_pattern.parent.glob(matte_pattern.name)),
        mapper=mapper,
        runner=runner,
    )


if __name__ == "__main__":
    from pysnptools.util.mapreduce1.runner import LocalMultiProc

    runner = LocalMultiProc(10)
    runa(runner)
    runb(runner)

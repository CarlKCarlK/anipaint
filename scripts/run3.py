import os
from pathlib import Path

from PIL import Image

from pysnptools.util.mapreduce1.runner import LocalMultiProc
from pysnptools.util.mapreduce1 import map_reduce


def run(runner):
    from anipaint import cached_edge_distance, paint

    folder = Path(r"E:\Dropbox\Watercolor Animation Assets")
    output_folder = folder / "SkinMatte/Comp 2/outputs/run3"

    matte_pattern = folder / "SkinMatte/Comp 2/*.*"
    cache_folder = folder / "SkinMatte/Comp 2/cache"

    brush_path = folder / "brushes/PaintStrokes (0-00-00-04).png"
    brush_image = Image.open(brush_path)

    os.makedirs(output_folder, exist_ok=True)

    def mapper(matte_path):
        print(matte_path)  # should log
        output_path = (output_folder / matte_path.name).with_suffix(".output.png")
        if output_path.exists():
            return
        edge_distance = cached_edge_distance(matte_path, cache_folder)

        im_in = paint(
            edge_distance,
            brush_image,
            random_count=250,
            keep_threshold=0.25,
            candidate_range=(5, 256),
            credit_range=(1, 256),
            mixing_range=(20, 75),
            sprite_factor=2,
        )

        im_edge = paint(
            edge_distance,
            brush_image,
            random_count=100,
            keep_threshold=0.1,
            candidate_range=(10, 11),
            credit_range=(1, 20),
            mixing_range=(255, 256),
            sprite_factor=2,
        )

        im_in.paste(im_edge, (0, 0), im_edge)
        im_in.save(output_path, optimize=True, compress_level=0)

    map_reduce(
        list(matte_pattern.parent.glob(matte_pattern.name)),
        mapper=mapper,
        runner=runner,
    )


if __name__ == "__main__":
    runner = LocalMultiProc(12)
    run(runner)

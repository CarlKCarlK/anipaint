from pathlib import Path

from pysnptools.util.mapreduce1.runner import LocalMultiProc

from anipaint import paint

runner = LocalMultiProc(10)

folder = Path(r"E:\Dropbox\Watercolor Animation Assets")
brush_pattern = folder / "brushes/*.png"

paint(
    output_folder=folder / "SkinMatte/Comp 2/outputs/run6a_4",
    matte_pattern=folder / "SkinMatte/Comp 2/*.*",
    brush_pattern=brush_pattern,
    random_count=500,
    outside_penalty=4,
    keep_threshold=0,
    candidate_range=(1, 256),
    credit_range=(1, 256),
    mixing_range=(255, 256),
    sprite_factor_range=(0.25, 1),
    runner=runner,
)

paint(
    output_folder=folder / "Comp 2/Comp 2/outputs/run6b_4",
    matte_pattern=folder / "Comp 2/Comp 2/*.*",
    brush_pattern=brush_pattern,
    random_count=500,
    outside_penalty=4,
    keep_threshold=0,
    candidate_range=(1, 256),
    credit_range=(1, 256),
    mixing_range=(255, 256),
    sprite_factor_range=(0.25, 1),
    runner=runner,
)

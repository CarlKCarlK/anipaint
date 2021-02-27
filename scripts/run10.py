import logging
from pathlib import Path

from pysnptools.util.mapreduce1.runner import LocalMultiProc

from anipaint import Paint

logging.basicConfig(level=logging.INFO)


frame_runner = LocalMultiProc(10)
batch_runner = None

folder = Path(r"E:\Dropbox\Watercolor Animation Assets")
brush_pattern = folder / "brushes/*.png"

Paint(
    output_folder=folder / "SkinMatte/Comp 2/outputs/run10",
    matte_pattern=folder / "SkinMatte/Comp 2/*.*",
    brush_pattern=brush_pattern,
    stroke_count_max=500,
    batch_count=50,
    penalty_area_pixels_max=30,
    brush_efficiency_min=None,
    candidate_range=(1, 256),
    credit_range=(1, 256),
    mixing_range=(255, 256),
    sprite_factor_range=(0.25, 1),
    frames_diff_fraction_max=None,
    frame_runner=frame_runner,
    batch_runner=batch_runner,
).paint()

Paint(
    output_folder=folder / "Comp 2/Comp 2/outputs/run10",
    matte_pattern=folder / "Comp 2/Comp 2/*.*",
    brush_pattern=brush_pattern,
    stroke_count_max=500,
    batch_count=50,
    penalty_area_pixels_max=30,
    brush_efficiency_min=None,
    candidate_range=(1, 256),
    credit_range=(1, 256),
    mixing_range=(255, 256),
    sprite_factor_range=(0.25, 1),
    frames_diff_fraction_max=0.01,
    frame_runner=frame_runner,
    batch_runner=batch_runner,
).paint()

import logging
from pathlib import Path

from anipaint import Paint

logging.basicConfig(level=logging.INFO)

# from pysnptools.util.mapreduce1.runner import LocalMultiProc
from pysnptools.util.mapreduce1.runner import LocalMultiThread

frame_runner = None  # LocalMultiProc(10)
batch_runner = LocalMultiThread(10)

folder = Path(r"m:\deldir\Watercolor Animation Assets")
brush_pattern = folder / "brushes/*.png"

Paint(
    output_folder=None,  # folder / "SkinMatte/Comp 2/outputs/run6a_4",
    matte_pattern=folder / "SkinMatte/Comp 2/Comp 2_00000.jpg",
    brush_pattern=folder / "brushes/*.png",
    stroke_count_max=500,
    batch_count=50,
    penalty_area_pixels_max=10,
    brush_efficiency_min=None,
    candidate_range=(1, 256),
    credit_range=(1, 256),
    mixing_range=(255, 256),
    sprite_factor_range=(0.25, 1),
    frame_runner=frame_runner,
    batch_runner=batch_runner,
).paint()[0].show()

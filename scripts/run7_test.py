import logging
from pathlib import Path

from anipaint import Paint

logging.basicConfig(level=logging.INFO)

# from pysnptools.util.mapreduce1.runner import LocalMultiProc
runner = None  # LocalMultiProc(10)

folder = Path(r"m:\deldir\Watercolor Animation Assets")
brush_pattern = folder / "brushes/*.png"

Paint(
    output_folder=None,  # folder / "SkinMatte/Comp 2/outputs/run6a_4",
    matte_pattern=folder / "SkinMatte/Comp 2/Comp 2_00000.jpg",
    brush_pattern=folder / "brushes/*.png",
    random_count=500,
    outside_penalty=4,
    keep_threshold=0,
    candidate_range=(1, 256),
    credit_range=(1, 256),
    mixing_range=(255, 256),
    sprite_factor_range=(0.25, 1),
    runner=runner,
).paint()[0].show()

import logging
from pathlib import Path

from pysnptools.util.mapreduce1.runner import LocalMultiProc
from pysnptools.util.mapreduce1.runner import LocalMultiThread

from anipaint import Paint

# This tells it to print info and warning messages
logging.basicConfig(level=logging.INFO)

# The number of processors to use, e.g. 10, 15, 20
frame_runner = LocalMultiProc(20)
preview_runner = LocalMultiThread(20)

# We set top_folder to the location of assets
top_folder = Path(r"E:\Dropbox\Watercolor Animation Assets")
brush_pattern = top_folder / "brushes/*.png"
# Instead, could have said:
#     brush_pattern =  Path(r"E:\Dropbox\Watercolor Animation Assets\brushes\*.png"

# Set to None to render all frames. Set to 0 to preview 1st frame
# set to 10 to preview 11th frame.
preview_frame = None  # 0

Paint(
    preview_frame=preview_frame,
    output_folder=top_folder / "SkinMatte/Comp 2/outputs/run10",
    matte_pattern=top_folder / "SkinMatte/Comp 2/*.*",
    brush_pattern=brush_pattern,
    stroke_count_max=500,
    batch_count=50,
    penalty_area_pixels_max=30,
    brush_efficiency_min=None,
    candidate_range=(1, 256),
    credit_range=(1, 256),
    mixing_range=(255, 256),
    sprite_factor_range=(0.25, 1.0),
    frames_diff_fraction_max=None,
    frame_runner=frame_runner,
    preview_runner=preview_runner,
).paint()

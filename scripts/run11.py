import logging
import os
from pathlib import Path

from pysnptools.util.mapreduce1.runner import LocalMultiProc, LocalMultiThread

from anipaint import Paint

logging.basicConfig(level=logging.INFO)

# We set top_folder to the location of assets
if os.environ.get("COMPUTERNAME") != "KADIE2":
    # Ben's computer
    processor_count = 20
    top_folder = Path(r"D:\STRANDING NEW\AnapaintAssets")
    brush_pattern = top_folder / "Brushes 1/*.png"
    frame_runner = LocalMultiProc(processor_count)
    preview_runner = LocalMultiThread(processor_count)
else:  # Carl's computer
    processor_count = 12
    top_folder = Path(r"M:\deldir\Watercolor Animation Assets")
    brush_pattern = top_folder / "BRUSHES/*.png"
    preview_runner = None  # LocalMultiThread(processor_count)
    frame_runner = LocalMultiProc(processor_count)

# Set to None to render all frames. Set to 0 to preview 1st frame
# set to 10 to preview 11th frame.
preview_frame = 0  #  # 0  # None  # 13  # None  # 0

Paint(
    preview_frame=preview_frame,
    output_folder=top_folder / "Tests/TheoInChair/output/Run7",
    matte_pattern=top_folder
    / "Tests/TheoInChair/OpeningCoversationTemp/ChairTalk_Pupils*.*",
    brush_pattern=brush_pattern,
    stroke_count_max=1000,
    batch_count=50,
    penalty_area_pixels_max=5,
    brush_efficiency_min=None,
    candidate_range=(1, 256),
    credit_range=(1, 256),
    mixing_range=(255, 256),
    sprite_factor_range=(0.1, 0.4),
    frames_diff_fraction_max=0.0000001,
    frame_runner=frame_runner,
    preview_runner=preview_runner,
).paint()

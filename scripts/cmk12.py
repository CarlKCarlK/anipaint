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
    # top_folder = Path(r"M:\deldir\Watercolor Animation Assets")
    top_folder = Path(r"E:\Dropbox\Watercolor Animation Assets")
    brush_pattern = top_folder / "BRUSHES/*.png"
    if True:
        frame_runner = LocalMultiProc(processor_count)
        preview_runner = LocalMultiThread(processor_count)
    else:
        frame_runner, preview_runner = None, None

# Set to None to render all frames. Set to 0 to preview 1st frame
# set to 10 to preview 11th frame.
preview_frame = None


# keep going if something goes wrong
Paint.batch(
    # Theo_crouch_blue_preset(medium)*.png
    # Theo_crouch_red_preset(small)*.png
    # ...
    matte_pattern=top_folder / "SkinMatte/presettest/*/*.*",
    output_folder=top_folder / "SkinMatte/outputs/run1",
    preset_folder=top_folder / "presets",
    preview_frame=preview_frame,
    frame_runner=frame_runner,
    preview_runner=preview_runner,
)

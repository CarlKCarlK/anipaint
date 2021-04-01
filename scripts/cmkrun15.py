import logging
import os
from pathlib import Path

from pysnptools.util.mapreduce1.runner import LocalMultiProc, LocalMultiThread

from anipaint import Paint

logging.basicConfig(level=logging.INFO)

# We set top_folder to the location of assets
if os.environ.get("COMPUTERNAME") != "KADIE2":
    processor_count = 20
    top_folder = Path(r"D:\STRANDING NEW\AnapaintAssets")
    background_pattern = top_folder / "BGs/*.*"
    brush_pattern = top_folder / "Brushes 1/*.*"
    frame_runner = LocalMultiProc(processor_count)
    preview_runner = LocalMultiThread(processor_count)
else:
    processor_count = 12
    top_folder = Path(r"M:\deldir\Watercolor Animation Assets")
    brush_pattern = top_folder / "BRUSHES/*.png"
    preview_runner = LocalMultiThread(processor_count)
    frame_runner = LocalMultiProc(processor_count)

# Set to None to render all frames. Set to 0 to preview 1st frame
# set to 10 to preview 11th frame.
preview_frame = None  # 13  # None  # 0

Paint.batch(
    max_distance=10,  #
    brush_pattern=brush_pattern,  # Override preset
    background_pattern=top_folder / "BG textures/*.*",  # Override preset
    matte_pattern=top_folder / "Batches/Scene20/scene20_V6/*.*",
    output_folder=top_folder / "Batches/Scene20/output/md10",
    scale_height=None,
    preset_folder=top_folder / "presets",
    preview_frame=preview_frame,
    frame_runner=frame_runner,
    preview_runner=preview_runner,
)


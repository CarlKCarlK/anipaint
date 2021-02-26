from pathlib import Path

# from pysnptools.util.mapreduce1.runner import LocalMultiProc

from anipaint import find_skips

runner = None  # LocalMultiProc(10)

folder = Path(r"m:\deldir\Watercolor Animation Assets")

find_skips(matte_pattern=folder / "SkinMatte/Comp 2/*.jpg",)

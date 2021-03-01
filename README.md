AniPaint -- Paint animation mattes

Install & Run
=============

To set up a new machine

* Install miniconda: https://docs.conda.io/en/latest/miniconda.html 64-bit windows Python 3.8
* Open a Conda command window
* pip install pysnptools
* conda install pillow

To set up an AniPaint command window

* Open a Conda command window
* cd /d to anipaint program folder (top level)
* set pythonpath=%cd%

To run a script:

* Start with an AniPaint command window (with pythonpath set)
* cd /d TO_FOLDER_WITH_SCRIPTS
* python NAMEOFPROGRAM.PY

Creating a Script
=================

See scripts/sample.py

Notes:
* Python is very picky about using spaces (not tabs)
* Python is very picky about columns lining up
* It will not render an output that is already there
    * So, to get a fresh render, delete the output and aim it out new output folder.
* "top_folder" is just a convenience.
* File paths can use "/" or "\\", but if they use "\\" the string must start with an "r"
    * For example, r"E:\Dropbox\Watercolor Animation Assets")
* Use "preview_frame" to turn on/off preview vs save to file.
* It tries to center each stroke on an empty pixel
* A folder of cached of edge distances will be placed under the matte
  folder. To get fresh edge distances, delete the cache folder or
  put the updated mattes in a new folder.
* You can put more than one "Paint(...).paint()" command in a script
  to render multiple folders of matte files. (Could also write simple
  Python loop to run on multiple matte folders.)

Parameters:

* Folders and patterns for input and output
    * output_folder=top_folder / "SkinMatte/Comp 2/outputs/run10",
    * matte_pattern=top_folder / "SkinMatte/Comp 2/*.*",
    * brush_pattern=top_folder / "brushes/*.png",
    * background_pattern=top_folder / "BG textures/*.*",


*   stroke_count_max=500,
    * number of stokes to try to paint
*   batch_count=50,
    * number of stokes to paint at once
*   penalty_area_pixels_max=30,
    * How many pixels are allowed in the masked off area
*   background_matte_blur=3,
    * How much to blur the matte used to composite in the background
*   brush_efficiency_min=None,
    * Fraction between 0.0 and 1.0 telling how much of the brush
       must paint empty pixels. Prevents bush overlaps.
*   candidate_range=(1, 256),
    *  How far from the edge to center a new stroke. The first number is
    inclusive and should always be at least 1. The 2nd number is exclusive.
    Edge distances are only measured up to 255.
*   credit_range=(1, 256),
    * The range of edge distances (inclusive exclusive) to credit
    a brush stroke for efficiency.
*   mixing_range=(255, 256),
    * From 1 to the 1st number (exclusive) will orient with the edge.
    * From the 2nd number (inclusive) will orient with the default angle.
    * From 1st (inclusive) to 2nd (exclusive), will do a linear
    weighted average of the angles.
*   sprite_factor_range=(0.25, 1.0),
    * Randomly scales the brush to this range.
*   frames_diff_fraction_max=None,
    * If None, every frame gets it own paint. If a fraction
    between 0.0 and 1.0 may paint a frame the same as its predecessor.
    For example, if .01 then frames that differ by less than 1% will get
    the same paint.

*   default_angle_degrees = 15
*   default_angle_sd = 5
    * standard deviation of randomness to add (subtract) from default angle.
*   frame_runner = None
    * How to render non-preview frames on multiple processors. None means
    use only one processor.    
*   preview_runner = None
    * When previewing, how to paint multiple stokes on multiple
      processors. None means use only one processor.
*   cache_folder = None
    * Folder for edge distance cache. None means a subfolder
    under the matte folder called "cache"
*   seed = 231
     * A number that controls the randomness.
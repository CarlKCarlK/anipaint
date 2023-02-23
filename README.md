# AniPaint -- Paint animation mattes

## Background

Anipaint is a Python program created by Carl & Ben to paint animation mattes.

The code is currently proprietary. We're currently sharing it with our friends.

It should run PC, Mac, or Linux. We run it on a PC. Carl can test on Linux. We’ve never tested it on a Mac.

It is a “command-line interface (CLI)” "scripting" “batch program”. In other words, you create a text file containing a script. Then, on a command line (or terminal), you type a command to run your script. The script runs and produces text and files as output. AniPaint has no graphical user interface (GUI) because programming GUIs takes 10 times more time.

## One-Time Setup

### To set up a new machine

* [Install Miniconda Python](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
* [Install Git](https://git-scm.com/download/)
* Open a Conda command window
  * On Windows, you press the Windows button and type “Anaconda Prompt”
* conda create --yes --name anipaint python=3.8
  * This creates a new environment called “anipaint” within Python 3.8
* conda activate anipaint
  * This tells Python to use the “anipaint” environment
* pip install git+https://github.com/CarlKCarlK/anipaint.git
  * This tells Python to install the AniPaint program from GitHub

To get lastest version from GitHub:
* pip install --upgrade git+https://github.com/CarlKCarlK/anipaint.git

## To run a script

* Get to the AniPaint command line
  * Open a Conda command window (On Windows, you press the Windows button and type “Anaconda Prompt")
  * conda activate anipaint
* Move to the folder with your script
  * Windows: cd /d FOLDER_WITH_SCRIPTS
  * Linux: cd FOLDER_WITH_SCRIPTS
  * Mac: cd FOLDER_WITH_SCRIPTS

* python NAMEOFSCRIPT.PY

## Creating a Script

See [scripts/sample.py](https://github.com/CarlKCarlK/anipaint/blob/main/scripts/sample.py)

Notes:

* Python is very picky about using spaces (not tabs)
* Python is very picky about columns lining up
* AniPaint will not render an output that is already there
  * So, to get a fresh render, delete the output and aim it out new output folder.
* "top_folder" is just a convenience.
* On Windows, file paths can use "/" or "\\", but if they use "\\" the string must start with an "r"
  * For example, r"E:\Dropbox\Watercolor Animation Assets")
* Use "preview_frame" to turn on/off preview vs save to file.
* It tries to center each stroke on an empty pixel
* A folder of cached of edge distances will be placed under the matte
  folder. To get fresh edge distances, delete the cache folder or
  put the updated mattes in a new folder.
* The cache folder also contains a text file containing the
  differences between pairs of files. For example, with name
  "Comp 2_00000.Comp 2_00141.diff_list.txt"
* You can put more than one "Paint(...).paint()" command in a script
  to render multiple folders of matte files. (Could also write simple
  Python loop to run on multiple matte folders.)

Parameters:

* Folders and patterns for input and output
  * output_folder=top_folder / "SkinMatte/Comp 2/outputs/run10",
  * matte_pattern=top_folder / "SkinMatte/Comp 2/*.*",
    * The main images
  * brush_pattern=top_folder / "brushes/*.png",
    * The brushes to be randomly selected
  * background_pattern=top_folder / "BG textures/*.*",
    * Used to fill in leftover whitespace

* preview_frame=None,
  * Set to None (default) to save to file, or to a frame number (starting at 0) to preview that frame.
* stroke_count_max=500,
  * number of stokes to try to paint
* batch_count=50,
  * number of stokes to paint at once
* penalty_area_pixels_max=30,
  * How many pixels are allowed in the masked off area
* background_matte_blur=3,
  * How much to blur the matte used to composite in the background
* brush_efficiency_min=None,
  * Fraction between 0.0 and 1.0 telling how much of the brush
       must paint empty pixels. Prevents bush overlaps.
* candidate_range=(1, 256),
  * How far from the edge to center a new stroke. The first number is
    inclusive and should always be at least 1. The 2nd number is exclusive.
    Edge distances are only measured up to 255.
* credit_range=(1, 256),
  * The range of edge distances (inclusive exclusive) to credit
    a brush stroke for efficiency.
* mixing_range=(255, 256),
  * From 1 to the 1st number (exclusive) will orient with the edge.
  * From the 2nd number (inclusive) will orient with the default angle.
  * From 1st (inclusive) to 2nd (exclusive), will do a linear
    weighted average of the angles.
* sprite_factor_range=(0.25, 1.0),
  * Randomly scales the brush to this range.
* frames_diff_fraction_max=None,
  * If None, every frame gets it own paint. If a fraction
    between 0.0 and 1.0 may paint a frame the same as its predecessor.
    For example, if .01 then frames that differ by less than 1% will get
    the same paint.

* default_angle_degrees = 15
* default_angle_sd = 5
  * standard deviation of randomness to add (subtract) from default angle.
* frame_runner = None
  * How to render non-preview frames on multiple processors. None means
    use only one processor.
* preview_runner = None
  * When previewing, how to paint multiple stokes on multiple
      processors. None means use only one processor.
* cache_folder = None
  * Folder for edge distance cache. None means a subfolder
    under the matte folder called "cache"
* seed = 231
  * A number that controls the randomness.

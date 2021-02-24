import logging
from pathlib import Path

# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    shared_datadir = Path(r"D:\OneDrive\programs\anipaint\anipaint\tests\data")
    tmp_path = Path(r"m:/deldir/anipaint/tests")

    # open a matte
    matte_file = "Comp 2/ShirtMAtte_00000.jpg"

    matte_path = shared_datadir / matte_file

    matte_image = Image.open(matte_path)
    # plt.imshow(matte_image)
    # plt.show()
    assert matte_image.mode == "RGB", f"Expect images to be RGB, not {matte_image.mode}"
    array0 = np.array(matte_image)
    print(np.unique(array0.mean(axis=2), return_counts=True))
    print("!!!cmk")
    # mask = array0.sum(axis=2) > mask_threshold

    # array0 -= object_speed
    # array0[~mask] = 0
    # array0 *= volume / np.abs(array0).max()

    # speed = (array0 * array0).sum(axis=2) ** 0.5
    # speed[~mask] = 0

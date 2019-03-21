import os
import sys
import numpy as np
from tqdm import tqdm
from imageio import imwrite

rd_path = "data/raw/"
sv_path = "data/img_ori/"
sv_ss_path = "data/img_ss/"
fn = "rand_000_ss.raw"
x_a = 64
x_b = 576
y_a = 224
y_b = 736

for filename in tqdm(os.listdir(rd_path)):
  if filename.endswith(".raw"):
    out_fn = filename.replace(".raw", ".png")
    sv_dir = sv_ss_path if "ss" in out_fn else sv_path
    sv_dir = sv_dir + out_fn
    with open(rd_path + filename) as fh:
        raw = fh.readlines()
        # Read the sizes of the image
        sizes = raw[0].split(" ")[:2]
        sizes = [int(size) for size in sizes]
        sizes.reverse()
        sizes.append(3)

        # Create a numpy array to store the image
        raw = raw[1:]
        raw = [pix.split(' ')[:3] for pix in raw]
        arr = [[int(rgb) for rgb in pix] for pix in raw]
        nparr = np.array(arr, dtype=np.uint8).reshape(sizes)
        nparr = nparr[x_a:x_b, y_a:y_b, :]
        # print(nparr.shape)
        imwrite(sv_dir, nparr)

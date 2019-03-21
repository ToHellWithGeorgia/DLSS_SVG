import os
import sys
from tqdm import tqdm

exec_dir = "../../build/drawsvg"
svg_dir = "data/rand_svg/"

for filename in tqdm(os.listdir(svg_dir)):
  if filename.endswith(".svg"):
    # print(filename)
    temp = svg_dir + filename
    cmd = exec_dir + " " + temp + " > /dev/null"
    os.system(cmd)

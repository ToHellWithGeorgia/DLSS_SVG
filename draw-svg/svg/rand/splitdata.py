import os
import sys
import random 
from shutil import copy2
from tqdm import tqdm

ori_dir = "data/img_ori/"
ss_dir = "data/img_ss/"
train_dir = "dataset/train/"
dev_dir = "dataset/dev/"
test_dir = "dataset/test/"

ori_imgs = os.listdir(ori_dir)
ss_imgs = os.listdir(ss_dir)
leng = len(ori_imgs)
random.shuffle(ori_imgs)

for idx, filename in tqdm(enumerate(ori_imgs)):
  file_pre = filename.replace(".png", "")
  ss_fn = file_pre + "_ss" + ".png"
  if ss_fn not in ss_imgs:
    continue

  # Train, dev, test = 7:2:1
  target_dir = ""
  if idx < 0.7 * leng:
    target_dir = train_dir
  elif idx < 0.9 * leng:
    target_dir = dev_dir
  else:
    target_dir = test_dir

  copy2(ori_dir + filename, target_dir + "ori/" + filename)
  copy2(ss_dir + ss_fn, target_dir + "ss/" + ss_fn)

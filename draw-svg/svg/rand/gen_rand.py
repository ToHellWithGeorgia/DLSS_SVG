import os
import sys
import random
from tqdm import tqdm

line1 = '<?xml version="1.0" encoding="utf-8"?>\n'
line2 = '<!-- Generator: Python script. InTeResTinG -->\n'
line3 = '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
line4 = '<svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" width="600.000px" height="600.000px" viewBox="0 0 600.000 600.000" enable-background="new 0 0 600.000 600.000" xml:space="preserve">\n'

# <polygon fill="#1865ED" points="0,213.637 123.343,0 246.686,213.637 "/>
poly_template = '<polygon fill="#color" points="coords "/>\n'

pre_lines = [line1, line2, line3, line4]
last_line = '</svg>'
file_pre = 'rand_'
file_post = '.svg'

# Parameters are here
save_dir = 'data/rand_svg/'
file_pre = save_dir + file_pre
file_start_idx = 1700
num_generated = 500
max_num_shapes = 70
min_num_shapes = 50
max_side = 600.0
min_side = 0.0

def str_pad_zero(num, leng=4):
  string = str(num)
  return '0' * (leng - len(string)) + string

def gen_rand_pairs(mini, maxi):
  ret = ''
  for _ in range(3):
    float1 = random.uniform(mini, maxi)
    float2 = random.uniform(mini, maxi)
    ret = ret + f'{float1:.3f}' + ',' + f'{float2:.3f}' + ' ' 
  return ret

def gen_rand_colors():
  r = str_pad_zero(hex(random.randint(0, 255))[2:], 2)
  g = str_pad_zero(hex(random.randint(0, 255))[2:], 2)
  b = str_pad_zero(hex(random.randint(0, 255))[2:], 2)
  return r + g + b

for idx in tqdm(range(num_generated)):
  file_idx = file_start_idx + idx
  fn = file_pre + str_pad_zero(file_idx) + file_post

  with open(fn, 'w') as fh:
    num_shapes = random.randint(min_num_shapes, max_num_shapes + 1)
    lines = pre_lines.copy()
    for _ in range(num_shapes):
      template = poly_template
      template = template.replace('coords', gen_rand_pairs(min_side, max_side))
      template = template.replace('color', gen_rand_colors())
      lines.append(template)
    lines.append(last_line)
    fh.writelines(lines)

import os, sys
import logging
import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from imageio import imread
from PIL import Image
import numpy as np
from torch.nn import functional as func
import random

def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.
    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.
    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.
        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_available_devices():
    """Get IDs of all available GPUs.
    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device('cuda:{}'.format(gpu_ids[0]))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids

class DLSSDataset(Dataset):
    def __init__(self, root_dir, transform=None, train_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train_transform = train_transform

        ori_dir = root_dir + 'ori/'
        ss_dir = root_dir + 'ss/'
        ori_imgs = os.listdir(ori_dir)
        ss_imgs = [img.replace(".png", "_ss.png") for img in ori_imgs]
        self.ori_imgs = [os.path.join(ori_dir, img) for img in ori_imgs]
        self.ss_imgs = [os.path.join(ss_dir, img) for img in ss_imgs]

    def __getitem__(self, idx):
        # ori = imread(self.ori_imgs[idx])
        # ss = imread(self.ss_imgs[idx])
        ori = Image.open(self.ori_imgs[idx])
        ss = Image.open(self.ss_imgs[idx])

        if self.train_transform:
            ori = self.train_transform(ori)

        if self.transform:
            ori = self.transform(ori)
            ss = self.transform(ss)

        return ori, ss

    def __len__(self):
        return len(self.ori_imgs)

# Add salt and pepper noise to image
def sap(img): 
    h,w = img.size 
    imgp = img.load()
    i = h * w // 40
    rd = random.sample(range(0,h*w), i) 
    rd_h = [num // w for num in rd] 
    rd_w = [num % w for num in rd] 
    for idx in range(i): 
        sap = int(random.randint(0,1) * 255) 
        imgp[rd_h[idx],rd_w[idx]] = (sap,sap,sap)
    
    return img

def LoG(img):
    weight = [
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -16, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0]
    ]
    weight = np.array(weight)

    weight_np = np.zeros((1, 1, 5, 5))
    weight_np[0, 0, :, :] = weight
    weight_np = np.repeat(weight_np, img.shape[1], axis=1)
    weight_np = np.repeat(weight_np, img.shape[0], axis=0)

    weight = torch.from_numpy(weight_np).type(torch.FloatTensor).to('cuda:0')

    return func.conv2d(img, weight, padding=1)

def HFEN(output, target):
    return torch.sum(torch.pow(LoG(output) - LoG(target), 2)) / torch.sum(torch.pow(LoG(target), 2))


def l1_norm(output, target):
    return torch.sum(torch.abs(output - target)) / torch.numel(output)
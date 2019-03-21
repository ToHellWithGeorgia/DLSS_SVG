import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from tqdm import tqdm
import os, sys
import random
import numpy as np

import util
from util import sap
from model import RecurrentAE
from args import get_train_args
import matplotlib.pyplot as plt

def loss_fn_rae(output, target):
    ls = util.l1_norm(output, target)
    lg = util.HFEN(output, target)
    return 0.875 * ls + 0.125 * lg, ls, lg

def main(args):
    args.save_dir = args.save_dir + "train/" + args.name + "/"
    dev_png_dir = args.save_dir + "dev_png/"
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(dev_png_dir)

    log = util.get_logger(args.save_dir, args.name)

    device, gpu_ids = util.get_available_devices()
    log.info("Using device: " + str(device))

    # Function to save images
    pil2tensor = transforms.ToTensor()
    tensor2pil = transforms.ToPILImage()

    # Set random seed
    log.info('Using random seed {}...'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get model
    log.info('Building model...')
    model = RecurrentAE(input_nc=3, device=device)
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99)) 

    # Get data loader
    log.info('Building dataset...')
    train_dataset = util.DLSSDataset(root_dir=args.train_dataset_dir,
                                     transform=transforms.Compose([
                                        transforms.CenterCrop(256),
                                        transforms.Lambda(sap),
                                        transforms.ToTensor()
                                     ]))   
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True)

    dev_dataset = util.DLSSDataset(root_dir=args.dev_dataset_dir,
                                   transform=transforms.Compose([
                                        transforms.CenterCrop(256),
                                        transforms.ToTensor(),
                                    ]))    
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    # Train rae
    log.info('Training')
    step = 0
    epoch = 0
    best_dev_loss = 1e9
    batch_size = args.batch_size
    dev_sv_idx = random.randint(0, 2)
    model.train()
    while epoch != args.num_epochs:
        epoch += 1
        # log.info('Starting epoch {}...'.format(epoch))

        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for ori, ss in train_loader:
                ori = ori.to(device)
                ss = ss.to(device)
                model = model.to(device)
                model.set_input(ori)
                model.reset_hidden()

                output = model()
                loss, ls, lg = loss_fn_rae(output, ss)
                loss_val = loss.item()
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch, Loss=loss_val)

        # Evaluate every epoch and save the best model
        model.eval()
        total_loss_val = 0
        for ori_dev, ss_dev in dev_loader:
            ori_dev = ori_dev.to(device)
            ss_dev = ss_dev.to(device)
            model = model.to(device)
            model.set_input(ori_dev)
            model.reset_hidden()

            output_dev = model()
            loss, ls, lg = loss_fn_rae(output_dev, ss_dev)
            total_loss_val += loss.item()
        if total_loss_val < best_dev_loss:
            best_dev_loss = total_loss_val
            torch.save(model.state_dict(), args.save_dir + "best_rae.pt")
            log.info("Better model saved. Average loss: " + \
                str(total_loss_val/len(dev_loader)))
        # Save the dev image for each epoch
        plt.imsave(dev_png_dir + "epoch_{}.png".format(epoch), \
            np.asarray(tensor2pil(output_dev[dev_sv_idx].cpu().detach())))

        if epoch == 1:
            plt.imsave(dev_png_dir + "origin.png", \
                       np.asarray(tensor2pil(ori_dev[dev_sv_idx].cpu().detach())))
            plt.imsave(dev_png_dir + "ss.png", \
                       np.asarray(tensor2pil(ss_dev[dev_sv_idx].cpu().detach())))
        
        if epoch % 15:
            torch.save(model.state_dict(), \
                       args.save_dir + "epoch_{}.pt".format(epoch))
        model.train()


if __name__ == '__main__':
    main(get_train_args())
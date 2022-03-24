#!/usr/bin/python
from __future__ import print_function
### python lib
import os, sys, argparse, glob, re, math, copy, pickle
from datetime import datetime
import numpy as np
### torch lib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
### custom lib
import networks
import datasets
import utils
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from loss import CharbonnierLoss
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatio-Temporal Alignment Network")
    ### model options
    parser.add_argument('-model',           type=str,     default="TransformNet",   help='TransformNet')
    parser.add_argument('-nf',              type=int,     default=32,               help='#Channels in conv layer')
    parser.add_argument('-blocks',          type=int,     default=5,                help='#ResBlocks')
    parser.add_argument('-norm',            type=str,     default='none',             choices=["BN", "IN", "none"],   help='normalization layer')
    parser.add_argument('-model_name',      type=str,     default='none',           help='path to save model')
    ### dataset options
    parser.add_argument('-datasets_tasks',  type=str,     default='W3_D1_C1_I1',    help='dataset-task pairs list')
    parser.add_argument('-data_dir',        type=str,     default='/media/xfang/Elements/codes/TIP_cascade4/data', help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to lists folder')
    parser.add_argument('-checkpoint_dir',  type=str,     default='checkpoints',    help='path to checkpoint folder')
    parser.add_argument('-crop_size',       type=int,     default=128,              help='patch size')
    parser.add_argument('-geometry_aug',    type=int,     default=1,                help='geometry augmentation (rotation, scaling, flipping)')
    parser.add_argument('-order_aug',       type=int,     default=1,                help='temporal ordering augmentation')
    parser.add_argument('-scale_min',       type=float,   default=0.5,              help='min scaling factor')
    parser.add_argument('-scale_max',       type=float,   default=2.0,              help='max scaling factor')
    parser.add_argument('-sample_frames',   type=int,     default=7,               help='#frames for training')
    ### loss optinos
    parser.add_argument('-alpha',           type=float,   default=50.0,             help='alpha for computing visibility mask')
    parser.add_argument('-loss',            type=str,     default="cb",             help="optimizer [Options: SGD, ADAM]")
    parser.add_argument('-w_Loss',           type=float,   default=1,               help='weight for L2 loss')
    # parser.add_argument('-VGGLayers',       type=str,     default="4",              help="VGG layers for perceptual loss, combinations of 1, 2, 3, 4")
    ### training options
    parser.add_argument('-solver',          type=str,     default="ADAM",           choices=["SGD", "ADAIM"],   help="optimizer")
    parser.add_argument('-momentum',        type=float,   default=0.9,              help='momentum for SGD')
    parser.add_argument('-beta1',           type=float,   default=0.9,              help='beta1 for ADAM')
    parser.add_argument('-beta2',           type=float,   default=0.999,            help='beta2 for ADAM')
    parser.add_argument('-weight_decay',    type=float,   default=0,                help='weight decay')
    parser.add_argument('-batch_size',      type=int,     default=6,                help='training batch size')
    parser.add_argument('-train_epoch_size',type=int,     default=1000,             help='train epoch size')
    parser.add_argument('-valid_epoch_size',type=int,     default=100,              help='valid epoch size')
    parser.add_argument('-epoch_max',       type=int,     default=601,              help='max #epochs')
    ### learning rate options
    parser.add_argument('-lr_init',         type=float,   default=1e-4,             help='initial learning Rate')
    parser.add_argument('-lr_offset',       type=int,     default=30,               help='epoch to start learning rate drop [-1 = no drop]')
    parser.add_argument('-lr_step',         type=int,     default=100,               help='step size (epoch) to drop learning rate')
    parser.add_argument('-lr_drop',         type=float,   default=0.25,              help='learning rate drop ratio')
    parser.add_argument('-lr_min_m',        type=float,   default=0.1,              help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')
    ###STAN
    parser.add_argument('--n_colors',       type=int,     default=3,                help='number of color channels to use')
    parser.add_argument('--n_resgroups',    type=int,     default=10,               help='number of residual groups')
    parser.add_argument('--n_resblocks',    type=int,     default=20,               help='number of residual blocks')
    parser.add_argument('--n_feats',        type=int,     default=64,               help='number of feature maps')
    parser.add_argument('--reduction',      type=int,     default=16,               help='number of feature maps reduction')
    parser.add_argument('--scale',          type=str,     default='2',              help='super resolution scale')
    parser.add_argument('--res_scale',      type=float,   default=1,                help='residual scaling')

    ### other options
    parser.add_argument('-seed',            type=int,     default=9487,             help='random seed to use')
    parser.add_argument('-threads',         type=int,     default=32,                help='number of threads for data loader to use')
    parser.add_argument('-suffix',          type=str,     default='',               help='name suffix')
    parser.add_argument('-cpu',             action='store_true',                    help='use cpu?')
    opts = parser.parse_args()
    ### adjust options
    opts.cuda = (opts.cpu != True)

    opts.lr_min = opts.lr_init * opts.lr_min_m
    ### default model name
    if opts.model_name == 'none':
        opts.model_name = "%s_B%d_nf%d_%s" %(opts.model, opts.blocks, opts.nf, opts.norm)
        opts.model_name = "%s_T%d_%s_pw%d_%sLoss_a%s_%s_lr%s_off%d_step%d_drop%s_min%s_es%d_bs%d" \
                %(opts.model_name, opts.sample_frames, \
                  opts.datasets_tasks, opts.crop_size, opts.loss, str(opts.alpha), \
                  opts.solver, str(opts.lr_init), opts.lr_offset, opts.lr_step, str(opts.lr_drop), str(opts.lr_min), \
                  opts.train_epoch_size, opts.batch_size)
   
    if opts.suffix != "":
        opts.model_name += "_%s" %opts.suffix
    opts.size_multiplier = 2 ** 2 ## Inputs to FlowNet need to be divided by 64
    print(opts)
    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed(opts.seed)
    ### model saving directory
    opts.model_dir = os.path.join(opts.checkpoint_dir, opts.model_name)  
    print("========================================================")
    print("===> Save model to %s" %opts.model_dir)
    print("========================================================")
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)
    ### initialize model
    print('===> Initializing model from %s...' %opts.model)
    model = networks.__dict__[opts.model](opts)
    ### initialize optimizer
    if opts.solver == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opts.lr_init, momentum=opts.momentum, weight_decay=opts.weight_decay)
    elif opts.solver == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=opts.lr_init, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2))
    else:
        raise Exception("Not supported solver (%s)" %opts.solver)
    ### resume latest model
    name_list = glob.glob(os.path.join(opts.model_dir, "model_epoch_*.pth"))     
    epoch_st = 0
    if len(name_list) > 0:
        epoch_list = []
        for name in name_list:
            s = re.findall(r'\d+', os.path.basename(name))[0]      
            epoch_list.append(int(s))
        epoch_list.sort()
        epoch_st = epoch_list[-1]    

    if epoch_st > 0:
        print('=====================================================================')
        print('===> Resuming model from epoch %d' %epoch_st)
        print('=====================================================================')
        ### resume latest model and solver
        model, optimizer = utils.load_model(model, optimizer, opts, epoch_st)    
    else:
        ### save epoch 0
        utils.save_model(model, optimizer, opts)
    print(model)
    num_params = utils.count_network_parameters(model)
    print('\n=====================================================================')
    print("===> Model has %d parameters" %num_params)
    print('=====================================================================')
    ### initialize loss writer
    loss_dir = os.path.join(opts.model_dir, 'loss')
    writer = SummaryWriter(loss_dir)
    opts.rgb_max = 1.0
    opts.fp16 = False
                                                        
    ### convert to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() >= 1:
        # print("torch.cuda.device_count()",torch.cuda.device_count())
        model = nn.DataParallel(model,device_ids=[0,1])
        model = model.to(device)
    model.train()
    # if isinstance(model,torch.nn.DataParallel):
	#     model = model.module
    ### create dataset
    train_dataset = datasets.MultiFramesDataset(opts, "train")  
    ### start training
    total_iter=0
    while model.module.epoch < opts.epoch_max:
        model.module.epoch += 1
        ### re-generate train data loader for every epoch
        data_loader = utils.create_data_loader(train_dataset, opts, "train")
        ### update learning rate
        current_lr = utils.learning_rate_decay(opts, model.module.epoch)  
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
 
        ### criterion and loss recorder
        if opts.loss == 'L2':
            criterion = nn.MSELoss(size_average=True).cuda()
        elif opts.loss == 'L1':
            criterion = nn.L1Loss(size_average=True).cuda()
        elif opts.loss == 'cb':
            criterion = CharbonnierLoss().cuda()
        else:
            raise Exception("Unsupported criterion %s" %opts.loss)
        ### start epoch
        ts = datetime.now()
        print('data_loader',len(data_loader))
        for iteration, batch in enumerate(data_loader, 1):                                    
            total_iter+=1
            ### convert data to cuda
            frame_i = []
            frame_p = []
            frame_i_temp = []
            frame_p_temp = []
            for t in range(opts.sample_frames): 
                frame_i.append(batch[t * 2].to(device))
                frame_p.append(batch[t * 2 + 1].to(device))  
            frame_o = []
            frame_o.append(frame_p[0]) ## first frame
            ### get batch time
            data_time = datetime.now() - ts
            ts = datetime.now()
            ### clear gradients
            optimizer.zero_grad()
            Loss = 0
            ### forward                              
            output_last_fea = None
            for t in range(opts.sample_frames):
                return_l = utils.index_generation(t, opts.sample_frames, 7, padding='reflection')
                frame_i1 = frame_i[return_l[0]]
                frame_i2 = frame_i[return_l[1]]
                frame_i3 = frame_i[return_l[2]]
                frame_i4 = frame_i[return_l[3]]
                frame_i5 = frame_i[return_l[4]]
                frame_i6 = frame_i[return_l[5]]
                frame_i7 = frame_i[return_l[6]]
                frame_p2 = frame_p[t]
                ### forward model
                output, out_fea = model(frame_i1,frame_i2,frame_i3,frame_i4, frame_i5,frame_i6, frame_i7,output_last_fea)
                output_last_fea = out_fea.detach()
                frame_o2 = output
                if opts.w_Loss > 0:
                    Loss += opts.w_Loss * criterion(frame_o2, frame_p2)

            overall_loss = Loss
            ### backward loss
            overall_loss.backward()
            ### update parameters
            optimizer.step()
            network_time = datetime.now() - ts
            ### print training info
            info = ""
            info += "Epoch %d; Batch %d / %d; " %(model.module.epoch, iteration, len(data_loader))
            info += "lr = %s; " %(str(current_lr))
            ## number of samples per second
            batch_freq = opts.batch_size / (data_time.total_seconds() + network_time.total_seconds())
            info += "data loading = %.3f sec, network = %.3f sec, batch = %.3f Hz\n" %(data_time.total_seconds(), network_time.total_seconds(), batch_freq)
            info += "\tmodel = %s\n" %opts.model_name
            ### print and record loss
            if opts.w_Loss > 0:
                writer.add_scalar('Loss', Loss.item(), total_iter)
                info += "\t\t%25s = %f\n" %("Loss", Loss.item())
            writer.add_scalar('Overall_loss', overall_loss.item(), total_iter)
            print(info)
        ### end of epoch
        ### save model
        utils.save_model(model.module, optimizer, opts)

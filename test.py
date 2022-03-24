#!/usr/bin/python
from __future__ import print_function
### python lib
import os, sys, argparse, glob, re, math, pickle, cv2, time
import numpy as np
### torch lib
import torch
import torch.nn as nn
### custom lib
import networks
import utils
import pdb
from torchvision import transforms
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"]="1"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spatio-Temporal Alignment Network')
    ### model options
    parser.add_argument('-method',          type=str,     required=True,            help='test model name')
    parser.add_argument('-epoch',           type=int,     required=True,            help='epoch')
    ### dataset options
    parser.add_argument('-dataset',         type=str,     required=True,            help='dataset to test')
    parser.add_argument('-phase',           type=str,     default="test",           choices=["train", "test"])
    parser.add_argument('-data_dir',        type=str,     default='/media/xfang/Elements/codes/TIP_cascade4/data',           help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to list folder')
    parser.add_argument('-checkpoint_dir',  type=str,     default='checkpoints',    help='path to checkpoint folder')
    parser.add_argument('-task',            type=str,     required=True,            help='evaluated task')
    parser.add_argument('-redo',            action="store_true",                    help='Re-generate results')
    ###RCAN
    parser.add_argument('--n_colors',       type=int,     default=3,                help='number of color channels to use')
    parser.add_argument('--n_resgroups',    type=int,     default=10,               help='number of residual groups')
    parser.add_argument('--n_resblocks',    type=int,     default=20,               help='number of residual blocks')
    parser.add_argument('--n_feats',        type=int,     default=64,               help='number of feature maps')
    parser.add_argument('--reduction',      type=int,     default=16,               help='number of feature maps reduction')
    parser.add_argument('--scale',          type=str,     default='4',              help='super resolution scale')
    parser.add_argument('--res_scale',      type=float,   default=1,                help='residual scaling')
    ### other options
    parser.add_argument('-cuda',             type=int,     default=True,                help='gpu device id')
    opts = parser.parse_args()
    opts.cuda = True
    opts.size_multiplier = 2 ** 2 ## Inputs to TransformNet need to be divided by 4
    print(opts)

    if opts.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without -cuda")
    ### load model opts
    opts_filename = os.path.join(opts.checkpoint_dir, opts.method, "opts.pth")
    print("Load %s" %opts_filename)
    with open(opts_filename, 'rb') as f:
        model_opts = pickle.load(f)
    ### initialize model
    print('===> Initializing model from %s...' %model_opts.model)
    model = networks.__dict__[model_opts.model](model_opts)
    ### load trained model
    model_list = sorted(glob.glob("./checkpoints/TransformNet_B5_nf32_none_T7_W3_D1_C1_I1_pw128_cbLoss_a50.0_ADAM_lr0.0001_off30_step100_drop0.25_min1e-05_es1000_bs6/model_epoch_*"))
    model_list = [fn for fn in model_list if os.path.basename(fn).endswith("pth")]
    psnr_stan_list = []
    for model_pth in model_list:
        print(model_pth)
        epoch = int(model_pth.split('epoch_')[-1].split('.pth')[0])
        model_filename = os.path.join(opts.checkpoint_dir, opts.method, "model_epoch_%d.pth" %opts.epoch)
        print("Load %s" %model_filename)
        state_dict = torch.load(model_filename)
        model.load_state_dict(state_dict['model'])
        ### convert to GPU
        device = torch.device("cuda" if opts.cuda else "cpu")

        model = model.to(device,)
        model.eval()
        ### load video list
        list_filename = os.path.join(opts.list_dir, "%s_%s.txt" %(opts.dataset, opts.phase))
        with open(list_filename) as f:
            video_list = [line.rstrip() for line in f.readlines()]
        times = []
        psnr_list = []
        ### start testing
        count = 0.0
        avg_psnr_stan = 0.0
        psnr_stan_all = 0.0
        psnr_bicub_all = 0.0

        for v in range(len(video_list)):
            video = video_list[v]
            print("Test %s on %s-%s video %d/%d: %s" %(opts.task, opts.dataset, opts.phase, v + 1, len(video_list), video))
            ## setup path
            input_dir = os.path.join(opts.data_dir, opts.phase, "input", opts.dataset, video)
            process_dir = os.path.join(opts.data_dir, opts.phase, "processed", opts.task, opts.dataset, video)
            output_dir = os.path.join(opts.data_dir, opts.phase, "output", opts.method, "epoch_%d" %opts.epoch, opts.task, opts.dataset, video)
            # output_dir = os.path.join("/media/xfang/Elements/codes/repeat/data", opts.phase, "output", opts.method, "epoch_%d" %opts.epoch, opts.task, opts.dataset, video)
     
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            #else:
                #print("Output frames exist, skip...")
                #continue
            frame_list = glob.glob(os.path.join(input_dir, "*.png"))
            output_list = glob.glob(os.path.join(output_dir, "*.png"))
            if len(frame_list) == len(output_list) and not opts.redo:
                print("Output frames exist, skip...")
                continue

            lstm_state = None
            psnr_dict = {}
            output_last_fea = None
            each_stan_psnr = 0.0
            each_list = []
            for t in range(len(frame_list)):
                count += 1
                ### load frames
                return_l = utils.index_generation(t, len(frame_list), 7, padding='reflection')
                
                frame_i1 = utils.read_img(os.path.join(input_dir, "%08d.png" %(return_l[0])))
                frame_i2 = utils.read_img(os.path.join(input_dir, "%08d.png" %(return_l[1])))
                frame_i3 = utils.read_img(os.path.join(input_dir, "%08d.png" %(return_l[2])))
                frame_i4 = utils.read_img(os.path.join(input_dir, "%08d.png" %(return_l[3])))
                frame_i5 = utils.read_img(os.path.join(input_dir, "%08d.png" %(return_l[4])))
                frame_i6 = utils.read_img(os.path.join(input_dir, "%08d.png" %(return_l[5])))
                frame_i7 = utils.read_img(os.path.join(input_dir, "%08d.png" %(return_l[6])))

                frame_p2 = utils.read_img(os.path.join(process_dir, "%08d.png" %(t)))

                with torch.no_grad():
                    ### convert to tensor
                    frame_i1 = utils.img2tensor(frame_i1).to(device)
                    frame_i2 = utils.img2tensor(frame_i2).to(device)
                    frame_i3 = utils.img2tensor(frame_i3).to(device)
                    frame_i4 = utils.img2tensor(frame_i4).to(device)
                    frame_i5 = utils.img2tensor(frame_i5).to(device)
                    frame_i6 = utils.img2tensor(frame_i6).to(device)
                    frame_i7 = utils.img2tensor(frame_i7).to(device)
                    frame_p2 = utils.img2tensor(frame_p2).to(device)
                    ### model input
                    output, out_fea = model(frame_i1, frame_i2,frame_i3, frame_i4, frame_i5,frame_i6, frame_i7,output_last_fea)  
                    
                    output_last_fea = out_fea.detach()
                    ### forward
                    ts = time.time()
                    frame_o2 = output
                    te = time.time()
                    times.append(te - ts)
                ### convert to numpy array
                frame_o2 = utils.tensor2img(frame_o2)
                ### resize to original size
                # frame_o2 = cv2.resize(frame_o2, (W_orig, H_orig))
                ### save output frame
                output_filename = os.path.join(output_dir, "%08d.png" %(t))

                utils.save_img(frame_o2, output_filename)
                frame_p2 = utils.tensor2img(frame_p2)
                psnr_stan = utils.psnr(frame_o2, frame_p2)

                psnr_stan_all += psnr_stan
                print("PSNR: stan %f" %  psnr_stan)
                each_stan_psnr += psnr_stan
                if t == int(len(frame_list)-1):
                    print ("Each_PSNR",each_stan_psnr/len(frame_list))
                    each_list.append(each_stan_psnr/len(frame_list))
        if count == 0:
            print("PSNR_stan=",psnr_bicub_all) 
        else:
            avg_psnr_stan = psnr_stan_all/count
            print("PSNR_stan=", avg_psnr_stan)
        psnr_stan_list.append({epoch: avg_psnr_stan})
    print("psnr_stan_list=", psnr_stan_list)


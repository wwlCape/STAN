#!/usr/bin/env bash
#export cuda-path
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
#export cudnn path
# cudnn_version=/home/vision/wwl/software/cuda9.0_deb/cuda/include/
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
cudnn_version=/usr/local/cuda-8.0/cudnnv6/cuda/include
export LD_LIBRARY_PATH=$cudnn_version:$LD_LIBRARY_PATH
# source activate sliou-pytorch
python train.py -datasets_tasks W3_D1_C1_I1 


 

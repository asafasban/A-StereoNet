"""
MIT License

Copyright (c) 2020 linjc16

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import argparse
import torch
import numpy as np
import random
import os
from torch.backends import cudnn
import matplotlib.pyplot as plt
from Options import parse_opt
from Sovlers import get_solver

def showDispMap():
    fig = plt.figure()
    depthData = np.load(r'D:\TartanDatasetRepository\TartanDataset\train\carwelding\Hard\P002\depth_left\000000_left_depth.npy')
    focal_length = 320.0
    baseline = 0.25
    mask = (depthData > 0.01) * (depthData < 20)
    disp = np.zeros_like(depthData)
    disp[mask] = focal_length * baseline / depthData[mask]
    ax1 = fig.add_subplot(1, 1, 1)
    disp *= (1.0 / disp.max()) # norm 0 - 1
    stacked_img = np.stack((disp,) * 3, axis=-1)
    ax1.imshow(stacked_img)
    plt.show()
    exit(150000)

def main():
    
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--options', type=str, help='Path to the option JSON file.', default='./Options/MyOwnTartan.json')
    args = parser.parse_args()
    opt = parse_opt(args.options)
    
    # GPU/CPU Specification.
    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_ids']
    os.environ['MKL_NUM_THREADS'] = opt['cpu_threads']
    os.environ['NUMEXPR_NUM_THREADS'] = opt['cpu_threads']
    os.environ['OMP_NUM_THREADS'] = opt['cpu_threads']
    
    # Deterministic Settings.
    if opt['deterministic']:
        torch.manual_seed(1000)
        np.random.seed(1000)
        random.seed(1000)
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True
    
    # Create solver.
    solver = get_solver(opt)
    
    # Run.
    solver.run()

if __name__ == "__main__":
    # showDispMap()
    main()

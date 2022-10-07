"""
MIT License

Copyright (c) 2022 SLAMcore

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
import pandas as pd

IMG_EXTENSIONS = [
    '.png', '.PNG'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_sequences(path):
    # Scenes (office, office2, carwelding, hospital)
    # Difficulty (Easy, Hard)
    # Sequences (P000, P001, ...)
    seqs = [os.path.join(path, scene, diff, seq) for scene in os.listdir(path) if os.path.isdir(os.path.join(path, scene)) for diff in os.listdir(os.path.join(path, scene)) if os.path.isdir(os.path.join(path, scene, diff)) for seq in os.listdir(os.path.join(path, scene, diff)) if os.path.isdir(os.path.join(path, scene, diff, seq)) ]

    return seqs

def read_tartanair(filepath):

    train_path = os.path.join(filepath, 'train')
    test_path = os.path.join(filepath, 'val')
    train_seqs = get_sequences(train_path)
    test_seqs = get_sequences(test_path)
    #train_seqs += test_seqs

    all_left_img = []
    all_right_img = []
    all_left_deps = []
    test_left_img = []
    test_right_img = []
    test_left_deps = []

    leftPrefix = 'image_left'
    rightPrefix = 'image_right'
    leftDepthPrefix = 'depth_left'

    for seq in train_seqs:
        leftDir = os.path.join(seq, leftPrefix)
        rightDir = os.path.join(seq, rightPrefix)
        depthLeftDir = os.path.join(seq, leftDepthPrefix)
        leftCnt = 0
        rightCnt = 0
        dispCnt = 0
        for img in os.listdir(leftDir):
            leftCnt+=1
            all_left_img.append(os.path.join(seq, leftPrefix, img))

        for img in os.listdir(rightDir):
            rightCnt+=1
            all_right_img.append(os.path.join(seq, rightDir, img))

        for img in os.listdir(depthLeftDir):
            dispCnt+=1
            all_left_deps.append(os.path.join(seq, depthLeftDir, img))

        if (leftCnt != rightCnt or leftCnt != dispCnt or rightCnt != dispCnt):
            print("diff detect for seq ", seq)

    for seq in test_seqs:
        leftDir = os.path.join(seq, leftPrefix)
        rightDir = os.path.join(seq, rightPrefix)
        depthLeftDir = os.path.join(seq, leftDepthPrefix)
        for img in os.listdir(leftDir):
            test_left_img.append(os.path.join(seq, leftPrefix, img))

        for img in os.listdir(rightDir):
            test_right_img.append(os.path.join(seq, rightDir, img))

        for img in os.listdir(depthLeftDir):
            test_left_deps.append(os.path.join(seq, depthLeftDir, img))

    print("Number of training images: ", len(all_left_img))
    print("Number of testing images: ", len(test_left_img))
    return all_left_img, all_right_img, all_left_deps, test_left_img, test_right_img, test_left_deps

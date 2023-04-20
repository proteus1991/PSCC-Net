"""
file: inpainting
about:
author: Xiaohong Liu
date: 03/10/20
"""

import sys

sys.path.append('./PythonAPI')
from pycocotools.coco import COCO
import numpy as np
import cv2
import imageio as io
import matplotlib.pyplot as plt
import pylab
import os
from PIL import Image
from PIL import ImageFilter
import argparse
import sys
import pdb

# args = parse_args()

dataDir = '../images'
dataType = 'train2014'
annFile = '../annotations/instances_{1}.json'.format(dataDir, dataType)

# initialize COCO api for instance annotations
coco = COCO(annFile)

imgIds = sorted(coco.getImgIds())

for idx in range(len(imgIds)):
    imgId = imgIds[idx]
    img_info = coco.loadImgs(imgId)[0]
    img = io.imread(os.path.join(dataDir, dataType, img_info['file_name']))    # image order BGR

    if img.ndim != 3:
        continue
    else:
        h, w, c = img.shape

    annsId = coco.getAnnIds(imgIds=imgId)
    anns = coco.loadAnns(annsId)

    if len(anns) == 0:
        print('no anns available')
        continue

    ann_index = np.random.randint(0, len(anns))
    mask = np.array(coco.annToMask(anns[ann_index]))

    # dilation
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask, kernel)

    index = 0
    while mask.sum() / (h*w) < 0.01 or mask.sum() / (h*w) > 0.5:
        ann_index = np.random.randint(0, len(anns))
        mask = np.array(coco.annToMask(anns[ann_index]))
        mask = cv2.dilate(mask, kernel)
        if index > 200:
            print('no suitable mask!')
            break
        index = index + 1

    if index > 200:
        continue

    img = Image.fromarray(img)
    img = img.resize((256, 256), resample=Image.BICUBIC)

    mask = Image.fromarray((mask * 255))
    mask = mask.resize((256, 256), resample=Image.BICUBIC)

    img = np.asarray(img)
    mask = np.asarray(mask)

    # im_inpainting = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)  # cv2.INPAINT_NS, cv2.INPAINT_TELEA

    mask_name = img_info['file_name'].replace('.jpg', '.png')

    io.imsave('./masks/{}'.format(mask_name), mask)
    io.imsave('./images/{}'.format(img_info['file_name']), img)
    # io.imsave('./fakes/{}.png'.format(idx), im_inpainting)
    print(idx)

print('finished')

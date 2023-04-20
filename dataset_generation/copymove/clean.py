"""
file: clean
about:
author: Xiaohong Liu
date: 30/09/20
"""

from api import USCISI_CMD_API
from matplotlib import pyplot
import os
import sys
import numpy as np
import imageio
from PIL import Image

lmdb_dir = './'
dataset = USCISI_CMD_API(lmdb_dir=lmdb_dir,
                         sample_file=os.path.join(lmdb_dir, 'samples.keys'),
                         differentiate_target=True)

total_num = dataset.nb_samples

for i in range(total_num):
    print(i)
    one_sample = dataset[i]
    image, tri_mask, _ = one_sample

    if image.ndim != 3:
        np.stack([image, image, image], axis=2)

    image = Image.fromarray(image)
    im_resized = image.resize((256, 256))

    mask = tri_mask[:, :, 0]
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0

    mask = mask.astype(np.uint8)

    mask = Image.fromarray(mask)
    mask_resized = mask.resize((256, 256))

    im_resized.save('fake/{0:05d}.png'.format(i))
    mask_resized.save('mask/{0:05d}.png'.format(i))

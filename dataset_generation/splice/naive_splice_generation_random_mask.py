"""
file: naive_splice_generation
about:
author: Xiaohong Liu
date: 12/10/20
"""

import os
from random import randrange, random
from PIL import Image
import imageio
import numpy as np
import cv2

def data_aug(img, data_aug_ind):
    if data_aug_ind == 0:
        return np.asarray(img)
    elif data_aug_ind == 1:
        return np.asarray(img.rotate(90, expand=True))
    elif data_aug_ind == 2:
        return np.asarray(img.rotate(180, expand=True))
    elif data_aug_ind == 3:
        return np.asarray(img.rotate(270, expand=True))
    elif data_aug_ind == 4:
        return np.asarray(img.transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 5:
        return np.asarray(img.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 6:
        return np.asarray(img.rotate(180, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 7:
        return np.asarray(img.rotate(270, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    else:
        raise Exception('Data augmentation index is not applicable.')


def translate(mask_image, translate_down, translate_up, size, ind=0):
    x = randrange(translate_down, translate_up)
    y = randrange(translate_down, translate_up)

    a = 1
    b = 0
    c = x  # left/right (i.e. 5/-5)
    d = 0
    e = 1
    f = y  # up/down (i.e. 5/-5)
    mask = mask_image.transform(tuple(size), Image.AFFINE, (a, b, c, d, e, f))

    assert list(mask.size) == size

    if 0.15 * size[0] * size[1] < np.sum(np.asarray(mask)) / 255 < 0.5 * size[0] * size[1] or ind > 20:
        return mask
    else:
        return translate(mask_image, translate_down, translate_up, size, ind=ind+1)


def rotate(mask_image, rotate_down, rotate_up, size, ind=0):
    theta = randrange(rotate_down, rotate_up)
    mask = mask_image.rotate(angle=theta)

    assert list(mask.size) == size

    if 0.15 * size[0] * size[1] < np.sum(np.asarray(mask)) / 255 < 0.5 * size[0] * size[1] or ind > 20:
        return mask
    else:
        return rotate(mask_image, rotate_down, rotate_up, size, ind=ind+1)


def scale(mask_image, scale_down, scale_up, size, ind=0):
    S = scale_down + (scale_up - scale_down) * random()
    mask = mask_image.resize([int(size[1]*S), int(size[0]*S)], resample=Image.BILINEAR)
    w, h = mask.width, mask.height
    if S > 1:
        left = (w - size[1]) / 2
        top = (h - size[0]) / 2
        right = (w + size[1]) / 2
        bottom = (h + size[0]) / 2
        mask = mask.crop((left, top, right, bottom))
    else:
        mask_np = np.asarray(mask)
        mask = np.zeros(size)
        left = (size[1] - w) // 2
        top = (size[0] - h) // 2
        mask[top:top + mask_np.shape[0], left:left + mask_np.shape[1]] = mask_np
        mask = Image.fromarray(mask)

    assert list(mask.size) == size

    if 0.15 * size[0] * size[1] < np.sum(np.asarray(mask)) / 255 < 0.5 * size[0] * size[1] or ind > 20:
        return mask
    else:
        return scale(mask_image, scale_down, scale_up, size, ind=ind+1)


def deformable(mask_image, deformable_down, deformable_up, size, ind=0):
    S = deformable_down + (deformable_up - deformable_down) * random()

    sdim = randrange(0, 2)

    if sdim == 0:
        mask = mask_image.resize([int(size[1]*S), size[0]], resample=Image.BILINEAR)
        w, h = mask.width, mask.height
        if S > 1:
            left = (w - size[1]) / 2
            top = (h - size[0]) / 2
            right = (w + size[1]) / 2
            bottom = (h + size[0]) / 2
            mask = mask.crop((left, top, right, bottom))
        else:
            mask_np = np.asarray(mask)
            mask = np.zeros(size)
            left = (size[1] - w) // 2
            top = (size[0] - h) // 2
            mask[top:top + mask_np.shape[0], left:left + mask_np.shape[1]] = mask_np
            mask = Image.fromarray(mask)
    else:
        mask = mask_image.resize([size[1], int(size[0]*S)], resample=Image.BILINEAR)
        w, h = mask.width, mask.height
        if S > 1:
            left = (w - size[1]) / 2
            top = (h - size[0]) / 2
            right = (w + size[1]) / 2
            bottom = (h + size[0]) / 2
            mask = mask.crop((left, top, right, bottom))
        else:
            mask_np = np.asarray(mask)
            mask = np.zeros(size)
            left = (size[1] - w) // 2
            top = (size[0] - h) // 2
            mask[top:top + mask_np.shape[0], left:left + mask_np.shape[1]] = mask_np
            mask = Image.fromarray(mask)

    assert list(mask.size) == size

    if 0.15 * size[0] * size[1] < np.sum(np.asarray(mask)) / 255 < 0.5 * size[0] * size[1] or ind > 20:
        return mask
    else:
        return deformable(mask_image, deformable_down, deformable_up, size, ind=ind+1)


if __name__ == '__main__':

    image_dir = '/put/your/path/here/'
    save_imdir = '/put/your/path/here/'
    save_madir = '/put/your/path/here/'

    num = 3

    size = [256, 256]

    # translate params
    translate_down = - 127
    translate_up = + 127

    # rotate params
    rotate_down = -180
    rotate_up = 180

    # scale params
    scale_down = 0.5
    scale_up = 2

    # deformable params
    deformable_down = 0.5
    deformable_up = 2

    imlist = []

    # image and mask lists
    with open(os.path.join(image_dir, 'alllist.txt')) as f:
        contents = f.readlines()
        for content in contents:
            if '_flat_' in content:
                continue
            else:
                imlist.append(content.strip())

    for n in range(num):

        print(n)

        copy_idx = randrange(0, len(imlist))
        paste_idx = randrange(0, len(imlist))

        index = 0

        while paste_idx == copy_idx:
            paste_idx = randrange(0, len(imlist))
            if index > 30:
                break
            index += 1

        copy_dir = os.path.join(image_dir, imlist[copy_idx])
        paste_dir = os.path.join(image_dir, imlist[paste_idx])

        # read images
        copy_image = Image.open(copy_dir)
        paste_image = Image.open(paste_dir)

        # resize images
        copy_image = copy_image.resize(size, resample=Image.BILINEAR)
        paste_image = paste_image.resize(size, resample=Image.BILINEAR)

        # augmentation
        copy_aug_idx = randrange(0, 8)
        copy_image = data_aug(copy_image, copy_aug_idx)

        paste_aug_idx = randrange(0, 8)
        paste_image = data_aug(paste_image, paste_aug_idx)

        # generate random masks
        mask_image = np.zeros([256, 256])

        rad = 0.2  # 0.2
        edgy = 0.01  # 0.05

        from curve_generation import get_random_points, get_bezier_curve

        a = get_random_points(n=7, scale=1)
        x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)

        x, y = x * 255, y * 255

        # contours = np.array([[50, 50], [50, 150], [150, 150], [150, 50]], dtype=np.int32)
        contours = np.array([(x[i], y[i]) for i in range(len(x))], dtype=np.int32)
        cv2.fillPoly(mask_image, [contours], 255)
        mask_image = cv2.GaussianBlur(mask_image, (5, 5), cv2.BORDER_DEFAULT)

        mask_image[mask_image > 127] = 255
        mask_image[mask_image <= 127] = 0

        mask_image = mask_image.astype(np.uint8)
        mask_image = Image.fromarray(mask_image)

        # resize masks
        mask_image = mask_image.resize(size, resample=Image.BILINEAR)

        # translate
        translate_idx = randrange(0, 2)
        if translate_idx == 1:
            mask_image = translate(mask_image, translate_down, translate_up, size)
            if np.sum(np.asarray(mask_image)) / 255 < 0.15 * size[0] * size[1] or np.sum(np.asarray(mask_image)) / 255 > 0.5 * size[0] * size[1]:
                print('translate are more than 50% or less than 15%')
                continue

        # rotate
        rotate_idx = randrange(0, 2)

        if rotate_idx == 1:
            mask_image = rotate(mask_image, rotate_down, rotate_up, size)
            if np.sum(np.asarray(mask_image)) / 255 < 0.15 * size[0] * size[1] or np.sum(np.asarray(mask_image)) / 255 > 0.5 * size[0] * size[1]:
                print('rotate are more than 50% or less than 15%')
                continue

        # scale
        scale_idx = randrange(0, 2)

        if scale_idx == 1:
            mask_image = scale(mask_image, scale_down, scale_up, size)
            if np.sum(np.asarray(mask_image)) / 255 < 0.15 * size[0] * size[1] or np.sum(np.asarray(mask_image)) / 255 > 0.5 * size[0] * size[1]:
                print('scale are more than 50% or less than 15%')
                continue

        # deformable
        deformable_idx = randrange(0, 2)

        if deformable_idx == 1:
            mask_image = deformable(mask_image, deformable_down, deformable_up, size)
            if np.sum(np.asarray(mask_image)) / 255 < 0.15 * size[0] * size[1] or np.sum(np.asarray(mask_image)) / 255 > 0.5 * size[0] * size[1]:
                print('deformable are more than 50% or less than 15%')
                continue

        mask_image = np.asarray(mask_image)
        copy_image = np.asarray(copy_image)
        paste_image = np.asarray(paste_image)

        mask = np.zeros_like(mask_image).astype(np.float32)

        mask[mask_image > 127] = 1
        mask[mask_image <= 127] = 0

        mask_c = np.stack([mask]*3, axis=2)

        spliced_image = mask_c * copy_image + (1 - mask_c) * paste_image

        save_name = imlist[copy_idx].split('/')[-1].split('.')[0] + '_' + imlist[paste_idx].split('/')[-1].split('.')[0] + '.tif'

        spliced_image = Image.fromarray(spliced_image.astype(np.uint8))
        spliced_image.save(os.path.join(save_imdir, save_name))

        imageio.imsave(os.path.join(save_madir, save_name.replace('.tif', '.png')), (mask * 255).astype(np.uint8))

































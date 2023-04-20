from __future__ import print_function
import os
import cv2
import json
import lmdb
import numpy as np
from matplotlib import pyplot


class USCISI_CMD_API(object):
    """ Simple API for reading the USCISI CMD dataset
    
    This API simply loads and parses CMD samples from LMDB

    # Example:
    ```python 
        # get the LMDB file path 
        lmdb_dir = os.path.dirname( os.path.realpath(__file__) )

        # create dataset instance
        dataset = USCISI_CMD_API( lmdb_dir=lmdb_dir, 
                                  sample_file=os.path.join( lmdb_dir, 'samples.keys'),
                                  differentiate_target=True )

        # retrieve the first 24 samples in the dataset
        samples = dataset( range(24) )
        # visualize these samples
        dataset.visualize_samples( samples )

        # retrieve 24 random samples in the dataset
        samples = dataset( [None]*24 )
        # visualize these samples
        dataset.visualize_samples( samples )
    
        # get the exact 50th sample in the dataset
        sample = dataset[50]
        # visualize these samples
        dataset.visualize_samples( [sample] )

    ```
    # Arguments:
        lmdb_dir = file path to the dataset LMDB
        sample_file = file path ot the sample list, e.g. samples.keys
        differentiate_target = bool, whether or not generate 3-class target map
    
    # Note:
        1. samples, i.e. the output of "get_samples" or "__call__", is a list of samples
        however, the dimension of each sample may or may not the same
        2. CMD samples are generated upon
           - MIT SUN2012 dataset [https://groups.csail.mit.edu/vision/SUN/]
           - MS COCO dataset [http://cocodataset.org/#termsofuse]
        3. detailed synthesis process can be found in paper
    
    # Citation:
        Yue Wu et.al. "BusterNet: Detecting Image Copy-Move ForgeryWith Source/Target Localization".  
        In: European Conference on Computer Vision (ECCV). Springer. 2018.

    # Contact:
        Dr. Yue Wu
        yue_wu@isi.edu
    """

    def __init__(self, lmdb_dir, sample_file, differentiate_target=True):
        assert os.path.isdir(lmdb_dir)
        self.lmdb_dir = lmdb_dir
        assert os.path.isfile(sample_file)
        self.sample_keys = self._load_sample_keys(sample_file)
        self.differentiate_target = differentiate_target
        print("INFO: successfully load USC-ISI CMD LMDB with {} keys".format(self.nb_samples))

    @property
    def nb_samples(self):
        return len(self.sample_keys)

    def _load_sample_keys(self, sample_file):
        '''Load sample keys from a given sample file
        INPUT:
            sample_file = str, path to sample key file
        OUTPUT:
            keys = list of str, each element is a valid key in LMDB
        '''
        with open(sample_file, 'r') as IN:
            keys = [line.strip() for line in IN.readlines()]
        return keys

    def _get_image_from_lut(self, lut):
        '''Decode image array from LMDB lut
        INPUT:
            lut = dict, raw decoded lut retrieved from LMDB
        OUTPUT:
            image = np.ndarray, dtype='uint8'
        '''
        image_jpeg_buffer = lut['image_jpeg_buffer']
        image = cv2.imdecode(np.array(image_jpeg_buffer).astype('uint8').reshape([-1, 1]), 1)
        return image

    def _get_mask_from_lut(self, lut):
        '''Decode copy-move mask from LMDB lut
        INPUT:
            lut = dict, raw decoded lut retrieved from LMDB
        OUTPUT:
            cmd_mask = np.ndarray, dtype='float32'
                       shape of HxWx1, if differentiate_target=False
                       shape of HxWx3, if differentiate target=True
        NOTE:
            cmd_mask is encoded in the one-hot style, if differentiate target=True.
            color channel, R, G, and B stand for TARGET, SOURCE, and BACKGROUND classes
        '''

        def reconstruct(cnts, h, w, val=1):
            rst = np.zeros([h, w], dtype='uint8')
            cv2.fillPoly(rst, cnts, val)
            return rst

        h, w = lut['image_height'], lut['image_width']
        src_cnts = [np.array(cnts).reshape([-1, 1, 2]) for cnts in lut['source_contour']]
        src_mask = reconstruct(src_cnts, h, w, val=1)
        tgt_cnts = [np.array(cnts).reshape([-1, 1, 2]) for cnts in lut['target_contour']]
        tgt_mask = reconstruct(tgt_cnts, h, w, val=1)
        if (self.differentiate_target):
            # 3-class target
            background = np.ones([h, w]).astype('uint8') - np.maximum(src_mask, tgt_mask)
            cmd_mask = np.dstack([tgt_mask, src_mask, background]).astype(np.float32)
        else:
            # 2-class target
            cmd_mask = np.maximum(src_mask, tgt_mask).astype(np.float32)

            # only output target mask
            # cmd_mask = tgt_mask.astype(np.float32)
        return cmd_mask

    def _get_transmat_from_lut(self, lut):
        '''Decode transform matrix between SOURCE and TARGET
        INPUT:
            lut = dict, raw decoded lut retrieved from LMDB
        OUTPUT:
            trans_mat = np.ndarray, dtype='float32', size of 3x3
        '''
        trans_mat = lut['transform_matrix']
        return np.array(trans_mat).reshape([3, 3])

    def _decode_lut_str(self, lut_str):
        '''Decode a raw LMDB lut
        INPUT:
            lut_str = str, raw string retrieved from LMDB
        OUTPUT: 
            image = np.ndarray, dtype='uint8', cmd image
            cmd_mask = np.ndarray, dtype='float32', cmd mask
            trans_mat = np.ndarray, dtype='float32', cmd transform matrix
        '''
        # 1. get raw lut
        lut = json.loads(lut_str)
        # 2. reconstruct image
        image = self._get_image_from_lut(lut)
        # 3. reconstruct copy-move masks
        cmd_mask = self._get_mask_from_lut(lut)
        # 4. get transform matrix if necessary
        trans_mat = self._get_transmat_from_lut(lut)
        return (image, cmd_mask, trans_mat)

    def get_one_sample(self, key=None):
        '''Get a (random) sample from given key
        INPUT:
            key = str, a sample key or None, if None then use random key
        OUTPUT:
            sample = tuple of (image, cmd_mask, trans_mat)
        '''
        return self.get_samples([key])[0]

    def get_samples(self, key_list):
        '''Get samples according to a given key list
        INPUT:
            key_list = list, each element is a LMDB key or idx
        OUTPUT:
            sample_list = list, each element is a tuple of (image, cmd_mask, )
        '''
        env = lmdb.open(self.lmdb_dir)
        sample_list = []
        with env.begin(write=False) as txn:         # txn: abbreviation of transaction
            for key in key_list:
                if not isinstance(key, str) and isinstance(key, int):
                    idx = key % self.nb_samples
                    key = self.sample_keys[idx]
                elif isinstance(key, str):
                    pass
                else:
                    key = np.random.choice(self.sample_keys, 1)[0]
                    print("INFO: use random key", key)
                lut_str = txn.get(key.encode())
                sample = self._decode_lut_str(lut_str)
                sample_list.append(sample)
        return sample_list

    def visualize_samples(self, sample_list):
        '''Visualize a list of samples
        '''
        for image, cmd_mask, trans_mat in sample_list:
            pyplot.figure(figsize=(10, 10))
            pyplot.subplot(121)
            pyplot.imshow(image)
            pyplot.subplot(122)
            pyplot.imshow(cmd_mask)
            pyplot.show()
        return

    def __call__(self, key_list):
        return self.get_samples(key_list)

    def __getitem__(self, key_idx):
        return self.get_one_sample(key=key_idx)

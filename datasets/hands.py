import os
import sys
import gzip
import struct
import requests

import numpy as np

import tensorflow as tf

from .datasets import ConditionalDataset

real_img_dir = 'real_imgs'
synthetic_img_dir = 'syn_imgs'

curdir = os.path.abspath(os.path.dirname(__file__))
outdir = os.path.join(curdir, 'files', 'hands')

CHUNK_SIZE = 32768


# def load_images(filename):
#     with gzip.GzipFile(filename, 'rb') as fp:
#         # Magic number
#         magic = struct.unpack('>I', fp.read(4))[0]
#
#         # item sizes
#         n, rows, cols = struct.unpack('>III', fp.read(4 * 3))
#
#         # Load items
#         data = np.ndarray((n, rows, cols), dtype=np.uint8)
#         for i in range(n):
#             sub = struct.unpack('B' * rows * cols, fp.read(rows * cols))
#             data[i] = np.asarray(sub).reshape((rows, cols))
#
#         return data


def load_data():
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # NEED TO PARSE EACH IMAGE!
    # x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
    # x_train = (x_train[:, :, :, np.newaxis] / 255.0).astype('float32')

    datasets = ConditionalDataset()
    datasets.real_image_dir = os.path.join(outdir, real_img_dir)
    datasets.synthetic_image_dir = os.path.join(outdir, synthetic_img_dir)
    datasets.real_images = os.listdir(datasets.real_image_dir)
    datasets.synthetic_images = os.listdir(datasets.synthetic_image_dir)
    datasets.attrs = None
    datasets.attr_names = [str(i) for i in range(1)]
    datasets.curr_image_batch = []

    return datasets

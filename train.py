import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf

from models import *
from datasets import load_data, mnist, svhn, hands

models = {
    'vae': VAE,
    'dcgan': DCGAN,
    'improved': ImprovedGAN,
    'resnet': ResNetGAN,
    'began': BEGAN,
    'wgan': WGAN,
    'lsgan': LSGAN,
    'cvae': CVAE,
    'cvaegan': CVAEGAN,
    'vaegan_mod': VAEGAN_mod,
    'vaegan': VAEGAN
}

def main(_):
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training GANs or VAEs')
    # parser.add_argument('--model', type=str, required=False, default='vae')
    parser.add_argument('--model', type=str, required=False, default='vaegan')
    # parser.add_argument('--dataset', type=str, required=False, default=r'mnist')
    parser.add_argument('--dataset', type=str, required=False, default=r'hands')
    # parser.add_argument('--dataset', type=str, required=False, default=r'.\datasets\files\celebA.hdf5')
    parser.add_argument('--num_channel', type=int, default=3)
    parser.add_argument('--save_period', type=int, default=10000)
    parser.add_argument('--output', default='output')
    parser.add_argument('--zdims', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--testmode', action='store_true')

    # Training parameters
    parser.add_argument('--datasize', type=int, default=-1)
    parser.add_argument('--lr_enc', type=float, default=2.0e-4)
    parser.add_argument('--lr_dec', type=float, default=2.0e-4)
    parser.add_argument('--lr_dis', type=float, default=2.0e-4)
    parser.add_argument('--beta1_enc', type=float, default=0.5)
    parser.add_argument('--beta1_dec', type=float, default=0.5)
    parser.add_argument('--beta1_dis', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--input_shape', type=int, default=32)
    parser.add_argument('--alpha_decay', type=float, default=0.7)
    parser.add_argument('--gamma_img2img', type=float, default=0.5)
    parser.add_argument('--gamma_dis', type=float, default=0.5)
    parser.add_argument('--use_feature_match', type=bool, default=False)

    args = parser.parse_args()

    # select gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Make output direcotiry if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    print('[*] Loading dataset')
    # Load datasets
    if args.dataset == 'mnist':
        datasets = mnist.load_data()
    elif args.dataset == 'svhn':
        datasets = svhn.load_data()
    elif args.dataset == 'hands':
        datasets = hands.load_data(args.datasize)
    else:
        datasets = load_data(args.dataset, args.datasize)

    # Construct model
    if args.model not in models:
        raise Exception('Unknown model:', args.model)

    if args.dataset == 'hands':
        input_size = (args.input_shape, args.input_shape, args.num_channel)
    else:
        input_size = datasets.shape[1:]

    print('[*] Initializing model')
    model = models[args.model](
        batchsize=args.batchsize,
        input_shape=input_size,
        attr_names=None or datasets.attr_names,
        z_dims=args.zdims,
        output=args.output,
        resume=args.resume,
        save_period=args.save_period,
        lr_enc=args.lr_enc,
        lr_dec=args.lr_dec,
        lr_dis=args.lr_dis,
        beta1_enc=args.beta1_enc,
        beta1_dec=args.beta1_dec,
        beta1_dis=args.beta1_dis,
        alpha_decay=args.alpha_decay,
        gamma_img2img=args.gamma_img2img,
        gamma_dis=args.gamma_dis,
        use_feature_match=args.use_feature_match)

    if args.testmode:
        model.test_mode = True

    tf.set_random_seed(12345)

    # Training loop
    if args.dataset != 'hands':
        datasets.images = datasets.images.astype('float32') * 2.0 - 1.0

    model.main_loop(datasets,
                    epochs=args.epoch)

if __name__ == '__main__':
    tf.app.run(main)
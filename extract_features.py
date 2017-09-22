import sys
import os.path
import argparse
import glob
import numpy as np
from scipy.misc import imread, imresize
import scipy.io
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
parser = argparse.ArgumentParser()
parser.add_argument('--model_def', type =str, default = 'models/VGG_ILSVRC_16_layers_deploy.prototxt', 
                    help='path to model definition prototxt')
parser.add_argument('--model', type = str, default = 'models/VGG_ILSVRC_16_layers.caffemodel', 
                    help='path to model parameters')
parser.add_argument('--gpu', type = bool, default = True, choices = ['True', 'False'], help='whether to use gpu')
parser.add_argument('--image_list', type = str, 
                    default = 'data/preprocessed/unique_images_train.txt', 
                    help='path to image_list text file')
parser.add_argument('--out_path', type = str, default = 'data/trainVGG.mat', help='path to output file. .mat file')


args = parser.parse_args()


import caffe

def predict(in_data, net):

    out = net.forward(**{net.inputs[0]: in_data})
    features = out[net.outputs[0]]
    return features


def batch_predict(filenames, net):

    N, C, H, W = net.blobs[net.inputs[0]].data.shape
    F = net.blobs[net.outputs[0]].data.shape[1]
    Nf = len(filenames)
    Hi, Wi, _ = imread(filenames[0]).shape
    allftrs = np.zeros((Nf, F))
    for i in range(0, Nf, N):
        in_data = np.zeros((N, C, H, W), dtype=np.float32)

        batch_range = range(i, min(i+N, Nf))
        batch_filenames = [filenames[j] for j in batch_range]
        Nb = len(batch_range)

        batch_images = np.zeros((Nb, 3, H, W))
        for j,fname in enumerate(batch_filenames):
            im = imread(fname)
            if len(im.shape) == 2:
                im = np.tile(im[:,:,np.newaxis], (1,1,3))
            # RGB -> BGR
            im = im[:,:,(2,1,0)]
            # mean subtraction
            im = im - np.array([103.939, 116.779, 123.68])
            # resize
            im = imresize(im, (H, W), 'bicubic')
            # get channel in correct dimension
            im = np.transpose(im, (2, 0, 1))
            batch_images[j,:,:,:] = im

        # insert into correct place
        in_data[0:len(batch_range), :, :, :] = batch_images

        # predict features
        ftrs = predict(in_data, net)

        for j in range(len(batch_range)):
            allftrs[i+j,:] = ftrs[j,:]

        print 'Done %d/%d files' % (i+len(batch_range), len(filenames))

    return allftrs


if args.gpu:
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()

net = caffe.Net(args.model_def, args.model, caffe.TEST)
base_dir = 'data'
unique_images = open(args.image_list, 'r').read().decode('utf8').splitlines()
files = []
for path in unique_images:
    files.append(os.path.join(base_dir, path))
allftrs = batch_predict(files, net)
allftrs = allftrs.astype(np.float32)
print('allftrs', allftrs.shape, allftrs.dtype)
scipy.io.savemat(args.out_path, mdict =  {'feats': np.transpose(allftrs)})

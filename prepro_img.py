import sys
import os
import argparse
import numpy as np
from scipy.misc import imread, imresize
import scipy.io
import caffe
import json
import h5py
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



INPUT_JSON = 'data_prepro.json'
IMAGE_ROOT = 'data'
CNN_PROTO = 'models/VGG_ILSVRC_16_layers_deploy.prototxt'
CNN_MODEL = 'models/VGG_ILSVRC_16_layers.caffemodel'
BATCH_SIZE = 64
OUT_NAME = 'data_img.h5'
GPU_LIST = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--input_json', type=str, default=INPUT_JSON, 
                    help='Path to the json file containing vocab and answers. Default: ' + INPUT_JSON + '.')
parser.add_argument('--image_root', type=str, default=IMAGE_ROOT, 
                    help='Path to image folder. Default: ' + IMAGE_ROOT + '.')
parser.add_argument('--cnn_proto', type=str, default=CNN_PROTO, 
                    help='Path to the cnn prototxt. Default: ' + CNN_PROTO + '.')
parser.add_argument('--cnn_model', type=str, default=CNN_MODEL, 
                    help='Path to the cnn model. Default: ' + CNN_MODEL + '.')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, 
                    help='Batch size. Default: ' + str(BATCH_SIZE) + '.')
parser.add_argument('--out_name', type=str, default=OUT_NAME, 
                    help='Output name. Default: ' + OUT_NAME + '.')
parser.add_argument('--gpu_list', type=str,
                    help='Which GPU to use. Default: ' + str(None) + ': No GPU.')

args = parser.parse_args()
print args
if args.gpu_list == None:
    caffe.set_mode_cpu()  
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list#"'" + str(args.gpu_id) + "'"
    caffe.set_mode_gpu()
    

net = caffe.Net(args.cnn_proto, args.cnn_model, caffe.TEST)
#print net

img_dir = args.image_root
data_train = json.load(open(args.input_json, 'r'))

train_list=[]
for i, file in enumerate(data_train['unique_img_train']):
    train_list.append(os.path.join(img_dir, file))
    
test_list = []
for i, file in enumerate(data_train['unique_img_test']):
     test_list.append(os.path.join(img_dir, file))

train_feat = batch_predict(train_list, net)
print ('train_feat', train_feat.shape)
test_feat = batch_predict(test_list, net)
print ('test_feat', test_feat.shape)

train_h5_file = h5py.File(args.out_name, "w")
train_h5_file['images_train'] = train_feat
train_h5_file['images_test'] = test_feat
train_h5_file.close()
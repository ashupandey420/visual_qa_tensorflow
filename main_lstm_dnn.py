import os
from tensorflow.python.client import device_lib
import tensorflow as tf
import argparse
from LSTM_DNN import LSTM_DNN
LR = 0.0002
BATCH_SIZE = 512
EPOCH = 300
NUM_LSTM_LAYER = 2
LSTM_LAYER_SIZE = 512
NUM_HIDDEN_LAYER = 2
HIDDEN_LAYER_SIZE = 1000
OUT_LAYER_SIZE = 1000
IMG_FEAT_SIZE = 1000
MAX_QUES_LENGTH = 26
WORD_EMBED_SIZE = 300
FINAL_FEAT_SIZE = 1024
GPUS = ''
IS_TRAIN = True
INPUT_IMG_H5 = 'data_img.h5'
INPUT_QUES_H5 = 'data_prepro.h5'
INPUT_JSON = 'data_prepro.json'
IS_TRAIN = True
TFRECORDS_PATH = 'data.tfrecords'
SAVE_DIR = 'model'
def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]
    
    
    parser = argparse.ArgumentParser(description='Adversarial Training DNN')
    parser.add_argument('--lr', type=float, default=LR, help='Learning rate for training. Default: ' + str(LR) + '.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--epoch', type=int, default=EPOCH, help='Number of Epoch to train. Default: ' + str(EPOCH) + '.')
    parser.add_argument('--num_lstm_layer', type=int, default=NUM_LSTM_LAYER, help='Number of LSTM layers '
                        'in question embedding model. Default: ' + str(NUM_LSTM_LAYER) + '.')
    parser.add_argument('--lstm_layer_size', type=int, default=LSTM_LAYER_SIZE, help='Size of LSTM layers '
                        'in question embedding model. Default: ' + str(LSTM_LAYER_SIZE) + '.')
    parser.add_argument('--num_hidden_layer', type=int, default=NUM_HIDDEN_LAYER, help='Number of hidden layers '
                        'in feed forward classifier. Default: ' + str(NUM_HIDDEN_LAYER) + '.')
    parser.add_argument('--hidden_layer_size', type=int, default=HIDDEN_LAYER_SIZE, help='Size of hidden layers '
                        'in feed forward classifier. Default: ' + str(HIDDEN_LAYER_SIZE) + '.')
    parser.add_argument('--out_layer_size', type=int, default=OUT_LAYER_SIZE, help='Number of classes in the '
                        'classifier. Default: ' + str(OUT_LAYER_SIZE) + '.')
    parser.add_argument('--img_feat_size', type=int, default=IMG_FEAT_SIZE, help='Size of feature used for image'
                        ' from the pre trained model. Default: ' + str(IMG_FEAT_SIZE) + '.')
    parser.add_argument('--max_ques_length', type=int, default=MAX_QUES_LENGTH, help='Maximum number of words allowed '
                        'in a question. Default: ' + str(MAX_QUES_LENGTH) + '.')
    parser.add_argument('--word_embed_size', type=int, default=WORD_EMBED_SIZE, help='Size of word embedding'
                        ' vector. Default: ' + str(WORD_EMBED_SIZE) + '.')
    parser.add_argument('--final_feat_size', type=int, default=FINAL_FEAT_SIZE, help='Size of final feature.'
                        ' Default: ' + str(FINAL_FEAT_SIZE) + '.')
    parser.add_argument('--gpus', type=str, default=GPUS, help='List of GPUS to use. Default: ' + str(GPUS) + '.')
    parser.add_argument('--input_img_h5', type=str, default=INPUT_IMG_H5, help='Input hdf5 file for image features.'
                        'Default: ' + str(INPUT_IMG_H5) + '.')
    parser.add_argument('--input_ques_h5', type=str, default=INPUT_QUES_H5, help='Input hdf5 file for question '
                        'and answer data. Default: ' + str(INPUT_QUES_H5) + '.')
    parser.add_argument('--input_json', type=str, default=INPUT_JSON, help='Input json file for question '
                        'and answer index lookup. Default: ' + str(INPUT_JSON) + '.')
    parser.add_argument('--is_train', type=_str_to_bool, default=IS_TRAIN, help='Whether to train. '
                        'Default: ' + str(IS_TRAIN) + ', Train the network')
    parser.add_argument('--tfrecords_path', type=str, default=TFRECORDS_PATH, help='Path to tfrecord file. '
                        'Default: ' + str(TFRECORDS_PATH) + '.')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR, help='Directory path where to save model parameters. '
                        'Default: ' + str(SAVE_DIR) + '.')
    args = parser.parse_args()
    return args

args = get_arguments()
print(args)
os.environ['CUDA_VISIBLE_DEVICES']=args.gpus
devices = device_lib.list_local_devices()      
udevices = [] 
for device in devices:
    if len(devices) > 1 and 'cpu' in device.name:
        # Use cpu only when we dont have gpus
        continue
    print('Using device: ', device.name)
    udevices.append(device.name)
    
print(udevices)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True

tf.reset_default_graph()
with tf.Session(config=config) as sess:
    if args.is_train:
        model = LSTM_DNN(sess, udevices, args, infer = False)
        model.train()
    
      

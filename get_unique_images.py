import numpy as np
import scipy.io
import sys
import operator
imdir='%s/COCO_%s_%012d.jpg'
questions_train = open('data/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
questions_lengths_train = open('data/preprocessed/questions_lengths_train2014.txt', 'r').read().decode('utf8').splitlines()
answers_train = open('data/preprocessed/answers_train2014_modal.txt', 'r').read().decode('utf8').splitlines()
images_train = open('data/preprocessed/images_train2014.txt', 'r').read().decode('utf8').splitlines()
images_train_path = open('data/preprocessed/images_train2014_path.txt', 'r').read().decode('utf8').splitlines()
answers_train_all = open('data/preprocessed/answers_train2014_all.txt', 'r').read().decode('utf8').splitlines()


questions_val = open('data/preprocessed/questions_val2014.txt', 'r').read().decode('utf8').splitlines()
questions_lengths_val = open('data/preprocessed/questions_lengths_val2014.txt', 'r').read().decode('utf8').splitlines()
answers_val = open('data/preprocessed/answers_val2014_modal.txt', 'r').read().decode('utf8').splitlines()
images_val = open('data/preprocessed/images_val2014_all.txt', 'r').read().decode('utf8').splitlines()
images_val_path = open('data/preprocessed/images_val2014_path.txt', 'r').read().decode('utf8').splitlines()
answers_val_all = open('data/preprocessed/answers_val2014_all.txt', 'r').read().decode('utf8').splitlines()
vgg_model_path = 'valVGG.mat'

def get_unique_img(img_ids, img_paths):
    count_img = {}
    N = len(img_ids)
    img_pos = np.zeros(N, dtype='uint32')
    for paths in img_paths:
        count_img[paths] = count_img.get(paths, 0) + 1

    unique_img = [w for w,n in count_img.iteritems()]
    imgtoi = {w:i for i,w in enumerate(unique_img)} 


    for i, path in enumerate(img_paths):
        img_pos[i] = imgtoi.get(path)

    return unique_img, img_pos

unique_img_train, img_pos_train = get_unique_img(images_train, images_train_path)
print(unique_img_train[:5])
print (img_pos_train[:5])
print ('unique_img_train', len(unique_img_train))
print ('img_pos_train', len(img_pos_train))
unique_img_file = 'data/preprocessed/unique_images_train.txt'
with open(unique_img_file, 'w') as f:
    for path in unique_img_train:
        f.write((path + '\n').encode('utf8'))
img_index_file = 'data/preprocessed/images_index_train.txt'
with open(img_index_file, 'w') as f:
    for pos in img_pos_train:
        f.write((str(pos) + '\n').encode('utf8'))
unique_img_file = 'data/preprocessed/unique_images_train.txt'
with open(unique_img_file, 'r') as f:
    u_im_tr = f.read().decode('utf8').splitlines()
    print ('u_im_tr', len(u_im_tr))
    print (u_im_tr[:5])
img_index_file = 'data/preprocessed/images_index_train.txt'
with open(img_index_file, 'r') as f:
    img_indices = f.read().decode('utf8').splitlines()
    print ('img_indices', len(img_indices))
    print img_indices[:5]
    
    
unique_img_val, img_pos_val = get_unique_img(images_val, images_val_path)
print(unique_img_val[:5])
print (img_pos_val[:5])
print ('unique_img_val', len(unique_img_val))
print ('img_pos_val', len(img_pos_val))
unique_img_file = 'data/preprocessed/unique_images_val.txt'
with open(unique_img_file, 'w') as f:
    for path in unique_img_val:
        f.write((path + '\n').encode('utf8'))
img_index_file = 'data/preprocessed/images_index_val.txt'
with open(img_index_file, 'w') as f:
    for pos in img_pos_val:
        f.write((str(pos) + '\n').encode('utf8'))
unique_img_file = 'data/preprocessed/unique_images_val.txt'
with open(unique_img_file, 'r') as f:
    u_im_val = f.read().decode('utf8').splitlines()
    print ('u_im_val', len(u_im_val))
    print (u_im_val[:5])
img_index_file = 'data/preprocessed/images_index_val.txt'
with open(img_index_file, 'r') as f:
    img_indices = f.read().decode('utf8').splitlines()
    print ('img_indices', len(img_indices))
    print img_indices[:5]
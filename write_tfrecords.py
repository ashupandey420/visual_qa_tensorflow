from collections import defaultdict
import numpy as np
import scipy.io
import sys
import operator
from sklearn.externals import joblib
from sklearn import preprocessing
from progressbar import Bar, ETA, Percentage, ProgressBar 
import tensorflow as tf
from nltk.tokenize import word_tokenize

def selectFrequentAnswers(questions_train, questions_lengths_train, answers_train, 
                          images_train, images_train_path, images_train_index,  
                          answers_train_all, max_answers):
    answer_fq= defaultdict(int)
    #build a dictionary of answers
    for answer in answers_train:
        answer_fq[answer] += 1

    sorted_fq = sorted(answer_fq.items(), key=operator.itemgetter(1), reverse=True)[0:max_answers]
    top_answers, top_fq = zip(*sorted_fq)
    new_questions_train = []
    new_questions_lengths_train = []
    new_answers_train = []
    new_images_train = []
    new_images_train_path = []
    new_images_train_index = []
    new_answers_train_all = []
    
    #only those answer which appear int he top 1K are used for training
    for question, question_length, answer, image, image_path, image_index, answer_all in zip(questions_train,
                                                                                     questions_lengths_train,
                                                                                     answers_train, 
                                                                                     images_train, 
                                                                                     images_train_path,
                                                                                     images_train_index, 
                                                                                     answers_train_all):
        if answer in top_answers:
            new_questions_train.append(question)
            new_questions_lengths_train.append(question_length)
            new_answers_train.append(answer)
            new_images_train.append(image)
            new_images_train_path.append(image_path)
            new_images_train_index.append(image_index)
            new_answers_train_all.append(answer_all)

    return (new_questions_train, new_questions_lengths_train,
            new_answers_train, new_images_train, 
            new_images_train_path, new_images_train_index, new_answers_train_all)

def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    print N
    for i in range(np.shape(seq)[0]):
        v[i, (N-lengths[i]):N]=seq[i, 0:lengths[i]]
    return v
def encode_questions(ques_tokens, wtoi, max_length):
    max_length = 25
    N = len(ques_tokens)
    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    for i, tokens in enumerate(ques_tokens):
        label_length[i] = min(max_length, len(tokens))
        for k, w in enumerate(tokens):
            if k < max_length:
                label_arrays[i,k] = wtoi[w]
    print label_arrays[0]
    label_arrays = right_align(label_arrays, label_length)
    
    return label_arrays

def get_tokens(sent_list):
    ans = []
    for sent in sent_list:
        ans.append(word_tokenize(sent.lower()))
    return ans
def final_tokens(tokens_list, counts, count_thr):
    ans = []
    for tokens in tokens_list:
        final_tokens = [w if counts.get(w,0) > count_thr else 'UNK' for w in tokens]
        ans.append(final_tokens)
    return ans

imdir='%s/COCO_%s_%012d.jpg'
questions_train = open('data/preprocessed/questions_train2014.txt', 'r').read().decode('utf8').splitlines()
questions_lengths_train = open('data/preprocessed/questions_lengths_train2014.txt', 'r').read().decode('utf8').splitlines()
answers_train = open('data/preprocessed/answers_train2014_modal.txt', 'r').read().decode('utf8').splitlines()
images_train = open('data/preprocessed/images_train2014.txt', 'r').read().decode('utf8').splitlines()
images_train_path = open('data/preprocessed/images_train2014_path.txt', 'r').read().decode('utf8').splitlines()
images_train_index = open('data/preprocessed/images_index_train.txt', 'r').read().decode('utf8').splitlines()
answers_train_all = open('data/preprocessed/answers_train2014_all.txt', 'r').read().decode('utf8').splitlines()
#get_unique_images_train()
print ('ques_train', len(questions_train), questions_train[0])
print ('ques_lengths', len(questions_lengths_train), questions_lengths_train[0])
print ('ans_train', len(answers_train), answers_train[0])
print ('imag_train', len(images_train), images_train[0])
print ('imag_train_path', len(images_train_path), images_train_path[0])
print ('imag_train_index', len(images_train_index), images_train_index[0])
print ('ans_train_all', len(answers_train_all), answers_train_all[0])
max_answers = nb_classes
questions_train, questions_lengths_train, answers_train, images_train, \
images_train_path, images_train_index, answers_train_all = selectFrequentAnswers(questions_train, 
                                                             questions_lengths_train, 
                                                             answers_train, 
                                                             images_train, 
                                                             images_train_path,
                                                             images_train_index, 
                                                             answers_train_all, 
                                                             max_answers)
#questions_lengths_train, questions_train, answers_train, answers_train_all, images_train = (list(t) for t in zip(*sorted(zip(questions_lengths_train, questions_train, answers_train, answers_train_all, images_train))))

questions_val = open('data/preprocessed/questions_val2014.txt', 'r').read().decode('utf8').splitlines()
questions_lengths_val = open('data/preprocessed/questions_lengths_val2014.txt', 'r').read().decode('utf8').splitlines()
answers_val = open('data/preprocessed/answers_val2014_modal.txt', 'r').read().decode('utf8').splitlines()
images_val = open('data/preprocessed/images_val2014_all.txt', 'r').read().decode('utf8').splitlines()
images_val_path = open('data/preprocessed/images_val2014_path.txt', 'r').read().decode('utf8').splitlines()
images_val_index = open('data/preprocessed/images_index_val.txt', 'r').read().decode('utf8').splitlines()
answers_val_all = open('data/preprocessed/answers_val2014_all.txt', 'r').read().decode('utf8').splitlines()
vgg_model_path = 'valVGG.mat'

#questions_lengths_val, questions_val, answers_val, images_val = (list(t) for t in zip(*sorted(zip(questions_lengths_val, questions_val, answers_val, images_val))))

ques_tokens_train = get_tokens(questions_train)
ques_tokens_val = get_tokens(questions_val)

counts = {}
count_thr = 5
for i, tokens in enumerate(ques_tokens_train):
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
print 'top words and their counts:'
print '\n'.join(map(str,cw[:20]))
# print some stats
total_words = sum(counts.itervalues())
print 'total words:', total_words
bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
vocab = [w for w,n in counts.iteritems() if n > count_thr]
bad_count = sum(counts[w] for w in bad_words)
print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
print 'number of words in vocab would be %d' % (len(vocab), )
print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)
print 'inserting the special UNK token'
vocab.append('UNK')



vocab_file = 'data/preprocessed/vocab_list_5.txt'
with open(vocab_file, 'w') as f:
    for word in vocab:
        f.write((word + '\n').encode('utf8'))
        
itow = {i+1:w for i,w in enumerate(vocab)} 
wtoi = {w:i+1 for i,w in enumerate(vocab)} 


ques_array_train = encode_questions(ques_tokens_train_final, wtoi, 25)
ques_array_val = encode_questions(ques_tokens_val_final, wtoi, 25)


labelencoder = preprocessing.LabelEncoder()
labelencoder.fit(answers_train)
nb_classes = len(list(labelencoder.classes_))

ans_array_train = labelencoder.transform(answers_train)
joblib.dump(labelencoder,'labelencoder_5.pkl')

vgg_model_path = 'data/trainVGG.mat'
features_struct = scipy.io.loadmat(vgg_model_path)
VGGfeatures = features_struct['feats']
print('VGGfeatures', VGGfeatures.shape)

img_array_train = VGGfeatures[:, map(int, images_train_index)]
img_array_train = img_array_train.T
img_array_train = preprocessing.normalize(img_array_train, 'l2', 1)
print('img_array_train', img_array_train.shape)

vgg_model_path = 'data/valVGG.mat'
features_struct = scipy.io.loadmat(vgg_model_path)
VGGfeatures = features_struct['feats']
print('VGGfeatures', VGGfeatures.shape)

img_array_val = VGGfeatures[:, map(int, images_val_index)]
img_array_val = img_array_val.T
img_array_val = preprocessing.normalize(img_array_val, 'l2', 1)
print('img_array_val', img_array_val.shape)

img_array_val = img_array_val.astype(np.float32)
ques_array_val = ques_array_val.astype(np.int32)
np.savez('val_data_5.npz', img_array_val = img_array_val, 
         ques_array_val = ques_array_val, answers_val_all = answers_val_all)

img_array_train = img_array_train.astype(np.float32)
ques_array_train = ques_array_train.astype(np.int32)
ans_array_train = ans_array_train.astype(np.int32)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecords(out_file, var_list, name_list):
    dict1 = {}
    for i in range(len(var_list)):
        dict1[name_list[i]] = _bytes_feature(var_list[i].tostring())
    example = tf.train.Example(features = tf.train.Features(feature = dict1))
    out_file.write(example.SerializeToString())
    
out_filepath = 'data_5.tfrecords'
if os.path.exists(out_filepath):
    os.unlink(out_filepath)
out_file = tf.python_io.TFRecordWriter(out_filepath)
for img, ques, ans in zip(img_array_train, ques_array_train, ans_array_train):
    write_tfrecords(out_file, [img, ques, ans], ['img', 'ques', 'ans'])
out_file.close()
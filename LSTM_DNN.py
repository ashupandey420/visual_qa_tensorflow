from model import Model
import timeit
import numpy as np
import tensorflow as tf
from ops import *
from itertools import cycle
from data_helpers import *
from tensorflow.contrib.layers import fully_connected
from model_helpers import *
from networks import QuestionEmbeddingNet, ImagePlusQuesFeatureNet, FeedForwardNet
from progressbar import Bar, ETA, Percentage, ProgressBar
import sys
from sklearn.externals import joblib
class LSTM_DNN(Model):
    def __init__(self, sess, devices, args, infer = False):
        super(LSTM_DNN, self).__init__('LSTM_DNN')
        self.sess = sess
        self.devices = devices
        self.epoch = args.epoch
        self.lr = tf.Variable(args.lr, name = 'lr', trainable = False)
        self.batch_size = args.batch_size
        self.num_lstm_layer = args.num_lstm_layer
        self.lstm_layer_size = args.lstm_layer_size
        self.num_hidden_layer = args.num_hidden_layer
        self.hidden_layer_size = args.hidden_layer_size
        self.img_feat_size = args.img_feat_size
        self.final_feat_size = args.final_feat_size
        self.max_ques_length = args.max_ques_length
        self.vocab_size = get_vocab_size(args.vocab_list)
        self.word_embed_size = args.word_embed_size
        self.out_layer_size = args.out_layer_size
        self.is_train = args.is_train
        self.save_path = args.save_dir
        self.results_path = args.result_path
        self.is_bnorm = args.is_bnorm
        self.feat_join = args.feat_join
        self.tfrecords_path = args.tfrecords_path
        self.labelencoder = joblib.load(args.lbl_enc_file)
        
        self.hidden_keep_prob = tf.Variable(args.hidden_keep_prob, dtype = tf.float32,
                                                trainable = False, 
                                                name = 'hidden_keep_prob')
        self.lstm_keep_prob = tf.Variable(args.lstm_keep_prob, dtype = tf.float32, 
                                              trainable = False, 
                                              name = 'lstm_keep_prob')
        self.is_vars_summ = True
        self.is_grads_summ = True
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.use_peepholes = args.use_peepholes
        self.ques_embed_net = QuestionEmbeddingNet(self.lstm_layer_size, 
                                                   self.num_lstm_layer, self.use_peepholes, self.is_bnorm)
        self.combine_feature =  ImagePlusQuesFeatureNet(self.final_feat_size, self.out_layer_size, 
                                                        tf.tanh, self.feat_join, self.is_bnorm)
        self.build_model(devices, args)
        self.train_writer, self.val_writer = self.summary_writer(sess.graph)
        self.merge_summaries()
        self.val_img, self.val_ques, self.val_ans = get_val_data(args.val_path)
        
        
       
    def build_model(self, devices, args):
        all_grads = []
        opt = self.optimizer
        print devices[0]
        with tf.variable_scope(self.name):
            for idx, device in enumerate(devices):
                with tf.device("/%s" % device):
                    with tf.name_scope("device_%s" % idx):
                        with variables_on_first_device(devices[0]):
                            self.build_model_single_gpu(idx, args)
                            grads = opt.compute_gradients(self.loss[-1],
                                                              var_list=self.vars)
                           
                            for grad in grads:
                                if grad[0] == None:
                                    print(grad[1].name)
                            
                            all_grads.append(grads)
                            tf.get_variable_scope().reuse_variables()
            avg_grads = average_gradients(all_grads)
            self.avg_grads = avg_grads
        self.opt = opt.apply_gradients(avg_grads)
     
        
    def build_model_single_gpu(self, idx, args):
        if idx == 0:
            self.img = []
            self.ques = []
            self.ans = []
            self.ques_embed = []
            self.final_img_feat = []
            self.final_ques_feat = []
            self.out_logit = []
            self.loss = []
            self.indata = read_data(self.tfrecords_path, 
                                    ['img', 'ques', 'ans'], 
                                    [(self.img_feat_size, ), (self.max_ques_length, ), ()], 
                                    [tf.float32, tf.int32, tf.int32])
            
        data = batch_data(self.indata, args.batch_size)
        self.img.append(data[0])
        self.ques.append(data[1])
        self.ans.append(data[2])
       
        if idx == 0:
            with tf.name_scope('Test_Model'):
                self.ques_embed_test = self.ques_embed_net(self.ques[idx], 
                                                      self.vocab_size, 
                                                      self.word_embed_size,
                                                      self.max_ques_length, 
                                                      self.batch_size, 
                                                      is_train = False, 
                                                      keep_prob = 1)
                self.ques_embed_W = self.ques_embed_net.ques_embed_W
                self.final_img_feat_test, self.final_ques_feat_test, \
                self.out_logit_test = self.combine_feature(self.img[idx], self.ques_embed_test, 
                                                           is_train = False, 
                                                           keep_prob = 1)
                self.out_proba_test = tf.nn.softmax(self.out_logit_test)
                self.val_accuracy = tf.placeholder(dtype = tf.float32, shape = ())
            tf.get_variable_scope().reuse_variables()
            print(tf.get_variable_scope())
        with tf.name_scope('Train_Model'):
            print(tf.get_variable_scope())
            ques_embed = self.ques_embed_net(self.ques[idx], 
                                                      self.vocab_size, 
                                                      self.word_embed_size,
                                                      self.max_ques_length, 
                                                      self.batch_size, 
                                                      is_train = True, 
                                                      keep_prob = self.lstm_keep_prob)
            final_img_feat, final_ques_feat, \
            out_logit = self.combine_feature(self.img[idx], ques_embed, 
                                              is_train = True, 
                                              keep_prob = self.hidden_keep_prob)
         
            self.ques_embed.append(ques_embed)
            self.final_img_feat.append(final_img_feat)
            self.final_ques_feat.append(final_ques_feat)
            self.out_logit.append(out_logit)
            if idx == 0:
                self.out_proba_train = tf.nn.softmax(self.out_logit[0])
            
           
        with tf.name_scope('cross_entropy_loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.out_logit[idx], 
                                                              labels = self.ans[idx]))
           
            self.loss.append(loss)
       
            
        self.get_vars()
        print('img_feat', self.img[idx].get_shape().as_list(), 
              self.img[idx].dtype)
        print('ques', self.ques[idx].get_shape().as_list(), 
              self.ques[idx].dtype)
        print('answer', self.ans[idx].get_shape().as_list(), 
              self.ans[idx].dtype)
        print('ques_embed', self.ques_embed[idx].get_shape().as_list(), 
              self.ques_embed[idx].dtype)
        print('ques_embed_W', self.ques_embed_W.get_shape().as_list(), 
              self.ques_embed_W.dtype)
        print('final_img_feat', self.final_img_feat[idx].get_shape().as_list(), 
              self.final_img_feat[idx].dtype)
        print('final_ques_feat', self.final_ques_feat[idx].get_shape().as_list(), 
              self.final_ques_feat[idx].dtype)
        print('out_logit', self.out_logit[idx].get_shape().as_list(), self.out_logit[idx].dtype)
        print('loss', self.loss[idx].get_shape().as_list(), self.loss[idx].dtype)
            
    def get_vars(self):
        vars1 = tf.trainable_variables()
        self.vars_dict = {}
        for var in vars1:
            print(var.name)
            self.vars_dict[var.name] = var
        self.vars = self.vars_dict.values()
             
    def merge_summaries(self):
        with tf.name_scope('summaries'):
            self.loss_summ = []
            for idx, loss in enumerate(self.loss):
                self.loss_summ.append(scalar_summary('device' + str(idx) + '/loss_summ', loss))
                
            self.vars_summ = []
            if self.is_vars_summ:
                for var in self.vars:
                    self.vars_summ.append(histogram_summary(var.name.replace(':','_'), var))
            self.grads_summ = []
            if self.is_grads_summ:
                for grad, var in self.avg_grads:
                    self.grads_summ.append(histogram_summary(var.name.replace(':','_') + '/gradients', grad))
            self.val_accuracy_summ = scalar_summary('val_accuracy',  self.val_accuracy)
            summ_lst = self.vars_summ + self.grads_summ + \
            self.loss_summ 
            if self.combine_feature.is_bnorm:       
                self.pop_mean_summ = []
                self.pop_var_summ = []
                self.pop_mean = self.combine_feature.batch_norm.pop_mean
                self.pop_var = self.combine_feature.batch_norm.pop_var
            
                for i in range(len(self.pop_mean)):
                    self.pop_mean_summ.append(histogram_summary(self.pop_mean[i].name, self.pop_mean[i]))
                    self.pop_var_summ.append(histogram_summary(self.pop_var[i].name, self.pop_var[i]))
                summ_lst += self.pop_mean_summ + self.pop_var_summ
            self.summ = tf.summary.merge(summ_lst)
    def summary_writer(self, graph):

        save_path = self.save_path
        if not os.path.exists(os.path.join(save_path, 'train')):
                os.makedirs(os.path.join(save_path, 'train'))
        if not os.path.exists(os.path.join(save_path, 'val')):
                os.makedirs(os.path.join(save_path, 'val'))
        if not os.path.exists(os.path.join(save_path, 'temp')):
                os.makedirs(os.path.join(save_path, 'temp'))
        train_writer = tf.summary.FileWriter(os.path.join(save_path,
                                                         'train'), graph)
        val_writer = tf.summary.FileWriter(os.path.join(save_path,
                                                         'val'))
        return train_writer, val_writer

    def train(self, model = None):
        devices = self.devices
        # change here to optimize in different ways
        opt = self.opt
        num_devices = len(devices)
        print('num_devices', num_devices)
        sess = self.sess
        init = tf.global_variables_initializer()
        sess.run(init)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        epoch = self.epoch
        num_examples = 0
        for record in tf.python_io.tf_record_iterator(self.tfrecords_path):
            num_examples += 1
        num_batches = num_examples / self.batch_size
        print('Number of examples: ', num_examples)
        print('Batches per epoch: ', num_batches)
        if self.load(self.save_path, model):
            print('[*] Load SUCCESS')
        else:
            print('[!] Load failed')
        counter = 0
        save_counter = 0
        batch_idx = 0
        curr_epoch = 0
        batch_timings = []
        losses = []
        try:
            while not coord.should_stop():
                start = timeit.default_timer()
                if batch_idx >= num_batches:
                    _, loss = sess.run([opt, self.loss])
                    losses.append(loss)
                        
                    batch_idx += num_devices
                    counter += num_devices
                    mean_losses = np.mean(losses, 0)
                    fdict = {}
                    for idx in range(num_devices):
                        fdict[self.loss[idx]] = mean_losses[idx]
                     
                    _summ = sess.run(self.summ, feed_dict = fdict) 
                    losses = []
                    curr_epoch += 1
                    batch_idx = 0
                    self.save(self.save_path, curr_epoch)
                    sess.run(self.increment_op)
                    self.train_writer.add_summary(_summ, sess.run(self.global_step))
                    self.train_writer.flush()
                    #val_accuracy_train = self.get_validation_score(1)
                    if (curr_epoch % 5) == 4:
                        val_accuracy_test = self.get_validation_score(0)
                        val_summ_test = sess.run(self.val_accuracy_summ, feed_dict = {self.val_accuracy:val_accuracy_test})
                        #val_summ_train = sess.run(self.val_accuracy_summ, feed_dict = {self.val_accuracy:val_accuracy_train})
                        self.val_writer.add_summary(val_summ_test, sess.run(self.global_step))
                        self.val_writer.flush()
                        #self.train_writer.add_summary(val_summ_train, sess.run(self.global_step))
                        #self.train_writer.flush()
                    
                    
                else:
                    _, loss = sess.run([opt, self.loss])
                    losses.append(loss)
                        
                    batch_idx += num_devices
                    counter += num_devices
                   
                end = timeit.default_timer()
                batch_timings.append(end - start)
                print('{}/{} (epoch {}), loss = {:.5f}, ' 
                      'time/batch = {:.3f}, '
                      'mtime/batch = {:.3f}'.format(counter,
                                                    epoch * num_batches,
                                                    curr_epoch,
                                                    np.mean(loss),
                                                    end - start,
                                                    np.mean(batch_timings)))
                
                if curr_epoch >= self.epoch:
                    # done training
                    print('Done training; epoch limit {} '
                          'reached.'.format(epoch))
                    print('Saving last model at epcoh {}'.format(curr_epoch))
                    self.save(self.save_path, self.global_step)
                    break

        except tf.errors.OutOfRangeError:
            print('Done training; epoch limit {} reached.'.format(self.epoch))
        finally:
            coord.request_stop()
        coord.request_stop()
        coord.join(threads)
        
        
        
    def get_validation_score(self, is_train = False):
        
        print('validation started')
        
        y_predict_text = []
        widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#',left='[',right=']'), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets)
        for qu_batch, an_batch, im_batch in pbar(zip(grouper(self.val_ques, self.batch_size, fillvalue=self.val_ques[-1]),
                                                   grouper(self.val_ans, self.batch_size, fillvalue=self.val_ans[-1]),
                                                   grouper(self.val_img, self.batch_size, fillvalue=self.val_img[-1]))):
            fdict = {}
            fdict[self.img[0]] = im_batch
            fdict[self.ques[0]] = qu_batch
            if is_train:
                y_proba = self.sess.run(self.out_proba_train, feed_dict = fdict)
            else:
                y_proba = self.sess.run(self.out_proba_test, feed_dict = fdict)
            y_predict = y_proba.argmax(axis = -1)
            y_predict_text.extend(self.labelencoder.inverse_transform(y_predict))
       

        correct_val = 0.0
        total = 0.
        binary_correct_val = 0.0
        binary_total = 0.1
        num_correct_val = 0.0
        num_total = 0.1
        other_correct_val = 0.0
        other_total = 0.1
        f1 = open(self.results_path, 'w')
        for prediction, truth, question, image in zip(y_predict_text, self.val_ans, self.val_ques, self.val_img):
            temp_count=0
            for _truth in truth.split(';'):
                if prediction == _truth:
                    temp_count+=1
            if temp_count>2:
                correct_val+=1
            else:
                correct_val+=float(temp_count)/3

            total += 1

            binary_temp_count = 0
            num_temp_count = 0
            other_count = 0
            if prediction == 'yes' or prediction == 'no':
                for _truth in truth.split(';'):
                    if prediction == _truth:
                        binary_temp_count+=1
                if binary_temp_count>2:
                    binary_correct_val+=1
                else:
                    binary_correct_val+= float(binary_temp_count)/3
                binary_total+=1
            elif np.core.defchararray.isdigit(prediction):
                for _truth in truth.split(';'):
                    if prediction == _truth:
                        num_temp_count+=1
                if num_temp_count>2:
                    num_correct_val+=1
                else:
                    num_correct_val+= float(num_temp_count)/3
                num_total+=1
            else:
                for _truth in truth.split(';'):
                    if prediction == _truth:
                        other_count += 1
                if other_count > 2:
                    other_correct_val += 1
                else:
                    other_correct_val += float(other_count) / 3
                other_total += 1

            #f1.write(question.encode('utf-8'))
            #f1.write('\n')
            #f1.write(image.encode('utf-8'))
            #f1.write('\n')
            f1.write(prediction)
            f1.write('\n')
            f1.write(truth.encode('utf-8'))
            f1.write('\n')
            f1.write('\n')

        f1.write('Final Accuracy is ' + str(correct_val/total))
        f1.close()
        f2 = open('overall_results.txt', 'a')
        f2.write(str(correct_val/total) + '\n\n')
        f2.write(str(binary_correct_val / binary_total) + '\n\n')
        f2.write(str(num_correct_val / num_total) + '\n\n')
        f2.write(str(other_correct_val / other_total) + '\n\n')
        f2.close()

        print 'Final Accuracy is', correct_val/total
        return correct_val/total
        
        
        

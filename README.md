# visual_qa_tensorflow
lstm_pipeline.sh has all the steps to be follwed from beginning.

Feature extraction is slow. You can keep already extracted features in data folder with names trainVGG.mat and valVGG.mat

download.sh downloads the questions and images also. Images are needed for feature extraction. If you have features then no need to download images.

pythondumpTxt.py has been taken from visual-qu Keras

get_unique_images.py computes the list of unique images which are to be used for feature extraction. It also dumps the question to image indices mapping in a text file. This indices are the indices of feature matrix which stores the features of each unique image.

write_tfrecords.py writes the trainign data in a tfrecord file which is read at the tim eof training. It also dumos the vocab list.

train_lstm.sh trains the model with default parameters. if you want to use gpu 0 then GPUS parameter will be '0'. For gpu 1 it will be '1'. For using both gpu 0 and 1 it will be '0, 1'. For no GPU use ''.

Currently data processing stage is hard coded so use the default file names as parameters.



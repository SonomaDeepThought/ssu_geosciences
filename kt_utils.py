import numpy as np
from PIL import Image

import os
import os.path

import shutil


import config # config file we need to load our params from


from tensorflow.python.client import device_lib

def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

def display_image(img, label):
        plt.imshow(img)
        print(label)
        plt.show()

def print_shapes(X_train, Y_train, X_dev, Y_dev, X_test, Y_test):
        print ("number of training examples = " + str(X_train.shape[0]))
        print ("number of dev examples = " + str(X_dev.shape[0]))
        print ("number of test examples = " + str(X_test.shape[0]))
        print ("X_train shape: " + str(X_train.shape))
        print ("Y_train shape: " + str(Y_train.shape))
        print ("X_dev shape : " + str(X_dev.shape))
        print ("Y_dev shape : " + str(Y_dev.shape))
        print ("X_test shape : " + str(X_test.shape))
        print ("Y_test shape : " + str(Y_test.shape))


def load_images(folder, img_size):
        '''
        return an array containing all the images from the given folder.
        all images are converted to RGB in channel_last format, and resized
        to img_size x img_size
        '''
        images = []
        for filename in os.listdir(folder):
                img = Image.open(os.path.join(folder, filename))
                if img is not None:
                        rbgimg = Image.new("RGB", img.size)
                        rbgimg.paste(img)
                        rbgimg = rbgimg.resize((img_size, img_size), Image.ANTIALIAS)
                        np_img = np.array(rbgimg)
                        images.append(np_img)

        return images


def load_subset(tenObject1, tenObject2, ratio, rangeStart1, rangeStart2, dRange1, dRange2):
        '''
        Args:
            tenObject1: Tensorflow object
            tenObject2: Tensorflow object
            ratio: desired images in the subset
	    rangeStart1: range of images used for set one
	    rangeStart2: range of images used for set two
	    dRange1: change in range for set one
	    dRange2: change in range for set two

        Returns: A dictionary containing the desired results.

        '''
        num_one = int(tenObject1.shape[0] * ratio)
        set_one = tenObject1[rangeStart1:num_one + dRange1]
        num_two = int(tenObject2.shape[0] * ratio)
        set_two = tenObject2[rangeStart2:num_two + dRange2]
        results = np.concatenate((set_one,set_two))
        return {'results': results, 'set_one': set_one, 'set_two' :set_two, 'num_one': num_one, 'num_two': num_two}


def load_dataset(image_directory, img_size, ratio_train = 0.6, ratio_dev = -1, ratio_test = -1,
                 verbose = False):

        assert(ratio_train > 0)
        assert(ratio_train < 1 and ratio_dev < 1 and ratio_test < 1)
        assert(ratio_train + ratio_dev + ratio_test <= 1)

        
        if ratio_dev < 0 and ratio_test < 0:
                ratio_dev = ratio_test = (1 - ratio_train) / 2
        elif ratio_dev < 0:
                ratio_dev = 1 - ratio_train - ratio_test
        elif ratio_test < 0:
                ratio_test = 1 - ratio_train - ratio_dev
                
        assert(ratio_train + ratio_dev + ratio_test == 1)

        dirs = os.listdir(image_directory)
        assert(len(dirs) == 2)
        
        x_one = np.array(load_images(image_directory + '/' + dirs[0], img_size))
        x_two = np.array(load_images(image_directory + '/' + dirs[1], img_size))
        y_one = np.ones((x_one.shape[0], 1))
        y_two = np.zeros((x_two.shape[0], 1))
        Y_images = np.concatenate((y_one, y_two))
        X_images = np.concatenate((x_one, x_two))
        
        # setup the training set
        x_train_all = load_subset(x_one, x_two, ratio_train, 0, 0, 0, 0)
        y_train_all = load_subset(y_one, y_two, ratio_train, 0, 0, 0, 0)
        
        # setup the dev set
        x_dev_all = load_subset(x_one, x_two, ratio_dev, x_train_all['num_one'], x_train_all['num_two'],
                                x_train_all['num_one'], x_train_all['num_two'])
        y_dev_all = load_subset(y_one, y_two, ratio_dev, y_train_all['num_one'], y_train_all['num_two'],
                                y_train_all['num_one'], y_train_all['num_two'])

        # setup the test set

        x_test_all = load_subset(x_one, x_two, ratio_test, x_train_all['num_one']+x_dev_all['num_one'],
                                 x_train_all['num_two'] + x_dev_all['num_two'], x_one.shape[0], x_two.shape[0])
        y_test_all = load_subset(y_one, y_two, ratio_test, y_train_all['num_one']+y_dev_all['num_one'],
                                 y_train_all['num_two'] + y_dev_all['num_two'], y_one.shape[0], y_two.shape[0])

        # retrieve results
        x_train = x_train_all['results']
        x_dev = x_dev_all['results']
        y_train = y_train_all['results']
        y_dev = y_dev_all['results']
        x_test = x_test_all['results']
        y_test = y_test_all['results']


        if verbose:
                print('train/dev/test ratios : ' + str(ratio_train) + ', ' + str(ratio_dev) + ', ' + str(ratio_test))
                print('x_images shape: ' + str(X_images.shape))
                print('y_images shape: ' + str(Y_images.shape))
                print('x_train shape: ' + str(x_train.shape))
                print('y_train shape: ' + str(y_train.shape))
                print('x_dev shape: ' + str(x_dev.shape))
                print('y_dev shape: ' + str(y_dev.shape))
                print('x_test shape: ' + str(x_test.shape))
                print('y_test shape: ' + str(y_test.shape))

        # check to ensure our sizes match
        assert(X_images.shape[0] == (x_train.shape[0] + x_dev.shape[0] + x_test.shape[0]))
        assert(Y_images.shape[0] == (y_train.shape[0] + y_dev.shape[0] + y_test.shape[0]))

        return x_train, y_train, x_dev, y_dev, x_test, y_test


def save_results(directory, model_name, history):
        '''
        copy the config.py file to directory/model_name/accuracy.txt
        append the results to the bottom of the file as a python block comment
        '''
        from shutil import copyfile
        filename = str(history.history['acc'][len(history.history['acc'])-1])
        copyfile('./config.py', directory + '/' + model_name + '/' +
                 filename + ".txt")
        f = open(directory + '/' + model_name + '/' +  filename + '.txt', 'a+')
        f.write('\'\'\'')
        f.write("\n\nresults: \n\n")
        for i in range(0, len(history.history['loss'])): 
                f.write('Epoch ' + str(i + 1) + '/' +
                        str(len(history.history['loss'])) + '\n')
                f.write('loss; ' + str(history.history['loss'][i]) + ' -  ')
                f.write('acc; ' + str(history.history['acc'][i]) + ' - \n')
                f.write('val_loss; ' +
                        str(history.history['val_loss'][i]) + ' - ')
                f.write('val_acc; ' +
                        str(history.history['val_acc'][i]) + '\n\n')

        f.write('\'\'\'')
        f.close()

def initialize_output_directory(directory, model_name):
        if not os.path.exists(directory):
                os.makedirs(directory)
        if not os.path.exists(directory + '/' + model_name):
                os.makedirs(directory + '/' + model_name)

def parse_config_file():
        loaded_params = {}
        loaded_params['model_name'] = config.model_name
        loaded_params['num_epochs'] = config.num_epochs
        loaded_params['batch_size'] = config.batch_size
        loaded_params['ratio_train'] = config.ratio_train
        loaded_params['ratio_test'] = config.ratio_test
        loaded_params['learning_rate'] = config.learning_rate
        loaded_params['output_directory'] = config.output_directory
        loaded_params['optimizer'] = config.optimizer
        loaded_params['image_directory'] = config.image_directory
        loaded_params['k_folds'] = config.k_folds
        gpus =  get_available_gpus()
        if gpus > config.num_gpus:
                loaded_params['num_gpus'] = config.num_gpus
        else:
                loaded_params['num_gpus'] = gpus
                print('Number of specified gpus in config.py exceeds available gpus')
        print("Using {", str(loaded_params['num_gpus']), "} gpus")
        return loaded_params

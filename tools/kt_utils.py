import numpy as np
from PIL import Image
from PIL import ImageFile
import scipy.misc
ImageFile.LOAD_TRUNCATED_IMAGES = True # allow truncated images to load 

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import os
import os.path

import shutil

import config # config file we need to load our params from

from tensorflow.python.client import device_lib


def print_cm(cm, labels, hide_zeroes=False,
             hide_diagonal=False, hide_threshold=None, output_file=None):
        """pretty print for confusion matrixes"""

        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = string = StringIO()
        
        columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
        empty_cell = " " * columnwidth
        # Print header
        print("    " + empty_cell, end=" ")
        for label in labels:
                print("%{0}s".format(columnwidth) % label, end=" ")
        print()
        # Print rows
        for i, label1 in enumerate(labels):
                print("    %{0}s".format(columnwidth) % label1, end=" ")
                for j in range(len(labels)):
                        cell = "%{0}.1f".format(columnwidth) % cm[i, j]
                        if hide_zeroes:
                                cell = cell if float(cm[i, j]) != 0 else empty_cell
                        if hide_diagonal:
                                cell = cell if i != j else empty_cell
                        if hide_threshold:
                                cell = cell if cm[i, j] > hide_threshold else empty_cell
                        print(cell, end=" ")
                print()
                
        sys.stdout = old_stdout
        print(string.getvalue())
        return string.getvalue()

                
def confusion_matrix(Y_true, Y_pred, labels=None, verbose=False):
        '''
        returns array, shape
        '''
        from sklearn.metrics import confusion_matrix

        # by defaults keras uses the round function to assign classes.
        # Thus the default threshold is 0.5

        Y_pred = np.rint(Y_pred[:,0])
        Y_true = Y_true[:,0]
        
        if verbose is True:
                print("Y_true: ", str(Y_true))
                print("Y_preds: ", str(Y_pred))

        return confusion_matrix(Y_true,Y_pred, labels=labels)
        

def get_available_gpus():
        '''
        Returns number of gpus on the current machine
        *Note : If this is called before setting tensorflow GPU, it will 
        force tensorflow onto the first gpu in the list
        '''
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
        filenames = []
        ignore_list = ['.AppleDouble', '__pycache__', '.DS_Store']
        for filename in os.listdir(folder):
                if filename in ignore_list:
                        continue
                img = Image.open(os.path.join(folder, filename))
                if img is not None:
                        rbgimg = Image.new("RGB", img.size)
                        rbgimg.paste(img)
                        rbgimg = rbgimg.resize((img_size, img_size), Image.ANTIALIAS)
                        np_img = np.array(rbgimg)
                        images.append(np_img)
                        filenames.append(filename)

        images = np.array(images)
        return images, filenames



def save_images(Y_pred, Y_actual, images, model_name,  ensembles=False):
        '''
        Y_preds: the unrounded predictions
        Y_actual: the labels
        images: the array containing all the images
        '''
        for i in range(0, images.shape[0]):

                conf = np.round(Y_pred, decimals=4)
                image_name = (model_name + "_" + str(i) +
                              "_confid_" +
                              str(conf[i]) + ".jpg")
                Y_i = np.round(Y_pred[i])
                if (Y_i - Y_actual[i]).all() :
                        # save image in correct folder
                        image_dir = "./classified_images/correct/"
                        if ensembles:
                                image_dir += "ensembles/"
                        save_image(image_dir + image_name, images[i])
                else:
                        # save image in incorrect folder
                        image_dir = "./classified_images/incorrect/"
                        if ensembles:
                                image_dir += "ensembles/"

                        save_image(image_dir + image_name,
                                   images[i])
                        
def save_image(file_path, image):
        scipy.misc.imsave(file_path, image)

def oversample(x, y, num_data_to_add):
        ''' 
        duplicate num_data_to_add number of elements from x and y 
        and return their concatenation
        '''

        batch_size = 32
        
        # we dont want to over sample if our sets are even
        if num_data_to_add <= 0 or num_data_to_add < batch_size:
                return x,y


        print("Oversampling images")
        datagen = ImageDataGenerator() # use no parameters to simply duplicate our images
        datagen.fit(x)
        p = x.copy()
        q = y.copy()
        num_added = 0
        for x_batch, y_batch in datagen.flow(x, y, batch_size=batch_size):
                p = np.concatenate((p, x_batch))
                q = np.concatenate((q, y_batch))
                num_added += x_batch.shape[0]
                if num_added >= num_data_to_add:
                        break
                        
        print("Num images added: ", str(num_added))

        return p,q

        
        

def data_augment(x, y, num_data_to_add, directory):
        '''
        Generate and append num_data_to_add.
        Return x,y as is if num_data_to_add is <= 0 
        '''

        batch_size = 1
        if num_data_to_add <= 0 or num_data_to_add < batch_size:
                return x, y


        print("Creating Augmented Images")
        datagen = ImageDataGenerator(
                    rotation_range=40, # degrees we can rotate max 180
                    width_shift_range=0.1, # percent width to shift
                    height_shift_range=0.1, # percent height to shift
                    shear_range=0.2, # angle in rotation ccw in radians
                    zoom_range=0.3,
                    horizontal_flip=True,
                    fill_mode='reflect')

        datagen.fit(x)
        q = y.copy()
        num_added = 0
        for x_batch, y_batch in datagen.flow(x, y, batch_size=batch_size,
                                             save_to_dir=directory,
                                             save_prefix='data_aug',
                                             save_format='jpeg'):
                if num_added == 0:
                        p = x_batch
                else :
                        p = np.concatenate((p,x_batch))
                q = np.concatenate((q, y_batch))
                num_added += x_batch.shape[0]
                if num_added >= num_data_to_add:
                        break
                        

        p = sort_aug_data(x, directory)
        return p,q


def sort_aug_data(x, directory):

        print('sort_aug_data: ' + str(x.shape) + ' ' + directory)
        
        import glob
        images = glob.glob(directory + "/*")
        images = sorted_nicely(images)
        img_size = x.shape[1]

        # sorted images from directory
        x_two = []
        for filename in images:
                img = Image.open(filename)
                if img is not None:
                        rbgimg = Image.new("RGB", img.size)
                        rbgimg.paste(img)
                        rbgimg = rbgimg.resize((img_size, img_size), Image.ANTIALIAS)
                        np_img = np.array(rbgimg)
                        x_two.append(np_img)

        
        x_two = np.array(x_two)

        import re

        indeces = [int(re.split('_', myImg)[3]) for myImg in images]

        x_prime = np.expand_dims(x[0], axis=0)
        idx = 0
        for i in range(0,x_two.shape[0]):
                if idx != int(indeces[i]):
                        idx += 1
                        if idx >= x.shape[0]:
                                break
                        x_prime = np.concatenate((x_prime,
                                                  np.expand_dims(x[idx],
                                                                 axis=0)))
                x_prime = np.concatenate((x_prime,
                                          np.expand_dims(x_two[i], axis=0)))
        x_prime = np.concatenate((x_prime, x[x_two.shape[0]:]))
        return x_prime

def sorted_nicely( l ):
        """ Sort the given iterable in the way that humans expect."""
        import re
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
        return sorted(l, key = alphanum_key)

def load_subset(tenObject1, tenObject2, ratio, rangeStart1,
                rangeStart2, dRange1, dRange2):
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


        TODO - Refactor to work with more than just binary input
        '''
        num_one = int(tenObject1.shape[0] * ratio)
        set_one = tenObject1[rangeStart1:num_one + dRange1]
        num_two = int(tenObject2.shape[0] * ratio)
        set_two = tenObject2[rangeStart2:num_two + dRange2]
        results = np.concatenate((set_one,set_two))
        return {'results': results, 'set_one': set_one, 'set_two' :set_two,
                'num_one': num_one, 'num_two': num_two}

def clean_dir(directory):
        remove_list = ['.AppleDouble', '__pycache__', '.DS_Store']
        for d in directory:
                if d in remove_list:
                        directory.remove(d)

        return directory

def load_generic_dataset(image_directory, img_size, ratio_train, ratio_dev,
                         ratio_test, use_data_augmentation, use_oversampling,
                         verbose, data_augment_directory):


        # oversampling takes precedence over data augmentation
        if use_oversampling:
                use_data_augmentation = False

        delta_size = 0

        dirs = os.listdir(image_directory)
        
        dirs = clean_dir(dirs)

        # get the number of elements in the largest directory.
        # this will be used to augment data up to the max
        max_num_files = -1
        for d in dirs:
                current = len(os.listdir(image_directory + '/' +  d))
                if current >= max_num_files:
                        max_num_files = current

        x_train_all = None
        y_train_all = None

        x_dev_all = None
        y_dev_all = None

        x_test_all = None
        y_test_all = None

        for i in range(0,len(dirs)):
                x,_ = load_images(image_directory + '/' + dirs[i], img_size)
                y = np.zeros((x.shape[0], 1))
                y[:] = i
                print('for dirs: ' + str(dirs[i]))
                print('x shape: ' + str(x.shape))
                # create x_test y_test first as to not modify it with data aug techniques
                # get the ranges for our images
                num_train = int(ratio_train*x.shape[0])
                num_dev = int(ratio_dev*x.shape[0])
                num_test = int(ratio_test*x.shape[0])

                x_train = x[:num_train,:]
                y_train = y[:num_train, :]

                print('initial x_train shape: ' + str(x_train.shape))
                
                x_dev = x[num_train:num_train+num_dev,:]
                y_dev = y[num_train:num_train+num_dev,:]

                x_test = x[num_train+num_dev:,:]
                y_test = y[num_train+num_dev:,:]


                current_num_files = y.shape[0]
                
                # only oversample training data
                if use_oversampling:
                       delta_size = max_num_files - current_num_files
                       x_train,y_train = oversample(x_train,y_train,delta_size)

                # only augment the training data
                if use_data_augmentation:
                        if not os.path.exists(data_augment_directory):
                                os.makedirs(data_augment_directory)
                        if not os.path.exists(data_augment_directory + '/' +
                                              dirs[i]):
                                os.makedirs(data_augment_directory + '/' +
                                            dirs[i])
                        
                                              
                        #current number of already augmented images
                        num_x_aug = len(next(os.walk(data_augment_directory +
                                                     '/' + dirs[i]))[2])

                        # add our augmented images to our image counters
                        current_num_files += num_x_aug

                        delta_size = max_num_files - current_num_files

                        #load our augmented images that already exist
                        print('loading augmented images... ')
                        x_aug, _ = load_images(data_augment_directory + '/' +
                                               dirs[i], img_size)

                        # found data to load
                        if x_aug.shape[0] != 0:
                                # add the data in aug directory to x_train
                                print('pre aug x_train shape' + str(x_train.shape))
                                x_train = sort_aug_data(x_train, data_augment_directory + '/' + dirs[i])
                                print('loaded ' + str(x_aug.shape[0]), 'augmented images')
                                print('post load x_train.shape ' + str(x_train.shape))

                        # regenerate our original labels with aug lables
                        y_train = np.zeros((x_train.shape[0],1))
                        y_train[:] = i

                        x_train, y_train = data_augment(x_train,
                                                        y_train,
                                                        delta_size,
                                                        data_augment_directory+
                                                        '/'+ dirs[i])


                if x_train_all is None:
                        x_train_all = x_train
                        y_train_all = y_train
                        x_dev_all = x_dev
                        y_dev_all = y_dev
                        x_test_all = x_test
                        y_test_all = y_test
                else:
                        print('concatenating')
                        x_train_all = np.concatenate((x_train_all, x_train))

                        print('y_train_All shape: ' + str(y_train_all.shape))
                        print('y_train shape: ' + str(y_train.shape))
                        y_train_all = np.concatenate((y_train_all, y_train))
                        x_dev_all = np.concatenate((x_dev_all, x_dev))
                        y_dev_all = np.concatenate((y_dev_all, y_dev))
                        x_test_all = np.concatenate((x_test, x_test))
                        y_test_all = np.concatenate((y_test_all, y_test))

                verbose = True
                if verbose:
                        print('x_train_all shape: ' + str(x_train_all.shape))
                        print('y_train_all shape: ' + str(y_train_all.shape))
                        print('x_dev_all shape: ' + str(x_dev_all.shape))
                        print('y_dev_all shape: ' + str(y_dev_all.shape))
                        print('x_test_all shape: ' + str(x_test_all.shape))
                        print('y_test_all shape: ' + str(y_test_all.shape))

        
        y_train_all = to_categorical(y_train_all, num_classes=len(dirs))
        y_dev_all = to_categorical(y_dev_all, num_classes=len(dirs))
        y_test_all = to_categorical(y_test_all, num_classes=len(dirs))

        print(y_train_all)
        
        
        return x_train_all, y_train_all, x_dev_all, y_dev_all, x_test_all, y_test_all, dirs
                        
                        


def load_dataset(image_directory, img_size, ratio_train = 0.6, ratio_dev = -1,
                 ratio_test = -1, use_data_augmentation = False, use_oversampling=False,
                 verbose = False, data_augment_directory=None):

        '''
        Generate x_train, y_train, x_dev, y_dev, x_test, y_test from images in the
        provided image directory

        TODO - refactor into something more readable. 
        TODO - Allow this function to work with more than just a binary set of data
        '''
        
        if ratio_dev < 0 and ratio_test < 0:
                ratio_dev = ratio_test = (1 - ratio_train) / 2
        elif ratio_dev < 0:
                ratio_dev = 1 - ratio_train - ratio_test
        elif ratio_test < 0:
                ratio_test = 1 - ratio_train - ratio_dev


        
        return load_generic_dataset(image_directory, img_size, ratio_train, ratio_dev, ratio_test, use_data_augmentation, use_oversampling, verbose, data_augment_directory)
                
        dirs = os.listdir(image_directory)

        # remove the mac generated files from the image directory
        dirs = clean_dir(dirs)

        # check that we have only 2 folders, since this is for binary class.
        assert(len(dirs) == 2)


        # oversampling takes precedence over data augmentaton
        if use_oversampling:
                use_data_augmentation = False
        
        delta_size_one = 0
        delta_size_two = 0


        # load our original images
        x_one,x_one_filenames = load_images(image_directory + '/' + dirs[0],
                                     img_size)
        x_two,x_two_filenames = load_images(image_directory + '/' + dirs[1],
                                     img_size)

        # generate our original labels
        y_one = np.ones((x_one.shape[0], 1))
        y_two = np.zeros((x_two.shape[0], 1))


        # create x_test y_test first as to not modify it with data aug techniques
        # get the ranges for our images
        num_train = int(ratio_train*x_one.shape[0])
        num_dev = int(ratio_dev*x_one.shape[0])
        num_test = int(ratio_test*x_one.shape[0])

        x_one_train = x_one[:num_train,:]
        y_one_train = y_one[:num_train, :]

        x_one_dev = x_one[num_train:num_train+num_dev, :]
        y_one_dev = y_one[num_train:num_train+num_dev, :]
     
        x_one_test = x_one[num_train+num_dev:, :]
        y_one_test = y_one[num_train+num_dev:, :]


        num_train = int(ratio_train*x_two.shape[0])
        num_dev = int(ratio_dev*x_two.shape[0])
        num_test = int(ratio_test*x_two.shape[0])

        x_two_train = x_two[:num_train,:]
        y_two_train = y_two[:num_train, :]

        
        x_two_dev = x_two[num_train:num_train+num_dev, :]
        y_two_dev = y_two[num_train:num_train+num_dev, :]
        
        
        x_two_test = x_two[num_train+num_dev:, :]
        y_two_test = y_two[num_train+num_dev:, :]


        num_x_one = x_one_train.shape[0]
        num_x_two = x_two_train.shape[0]
        
        if use_oversampling:
                delta_size_one = num_x_two - num_x_one
                delta_size_two = num_x_one - num_x_two

                # duplicate the larger delta into our image array
                x_one_train, y_one_train = oversample(x_one_train, y_one_train, delta_size_one)
                x_two_train, y_two_train = oversample(x_two_train, y_two_train, delta_size_two)


                
                
        if use_data_augmentation:
                # Generate the folders to store our augmented images if they
                # do not exist
                if not os.path.exists(data_augment_directory):
                        os.makedirs(data_augment_directory)
                if not os.path.exists(data_augment_directory + '/' + dirs[0]):
                        os.makedirs(data_augment_directory + '/' + dirs[0])
                if not os.path.exists(data_augment_directory + '/' + dirs[1]):
                        os.makedirs(data_augment_directory + '/' + dirs[1])

                        
                # count our number of augmented images that already exist
                '''
                TODO: This could be made more efficient
                '''
                num_x_aug_one = len(next(os.walk(data_augment_directory +
                                         '/' + dirs[0]))[2])
                num_x_aug_two = len(next(os.walk(data_augment_directory +
                                         '/' + dirs[1]))[2])
                
                # add our image counters
                num_x_one += num_x_aug_one
                num_x_two += num_x_aug_two
        
                # calculate the difference of our two image sets
                delta_size_one = num_x_two - num_x_one
                delta_size_two = num_x_one - num_x_two

                # load our augmented images from their directory
                print("Loading Augmented images...")
                x_one_aug, x_one_aug_filenames = load_images(data_augment_directory +
                                                 '/' + dirs[0],
                                                 img_size)
                x_two_aug, x_two_aug_filenames = load_images(data_augment_directory +
                                                 '/' + dirs[1],
                                                 img_size)

                
                # if we loaded an image print how many we loaded
                # append them to their correct collection
                if x_one_aug.shape[0] != 0:
                        print("Loaded ", str(x_one_aug.shape[0]), " augmented images")
                        x_one_train = sort_aug_data(x_one_train, data_augment_directory +
                                              '/' + dirs[0])
                        

                if x_two_aug.shape[0] != 0:
                        print("Loaded ", str(x_two_aug.shape[0]), " augmented images")
                        x_two_train = sort_aug_data(x_two_train, data_augment_directory +
                                              '/' + dirs[1])
                # regenerate our original labels with our aug labels included
                y_one_train = np.ones((x_one_train.shape[0], 1))
                y_two_train = np.zeros((x_two_train.shape[0], 1))

                        
                # Generate new images, append them to their collection
                # and write them to the correct folder
                x_one_train, y_one_train = data_augment(x_one_train, y_one_train,
                                            delta_size_one,
                                            data_augment_directory + '/' + dirs[0])
                x_two_train, y_two_train = data_augment(x_two_train, y_two_train,
                                            delta_size_two,
                                            data_augment_directory + '/' + dirs[1])
                        


        print("x_one shape: ", str(x_one.shape))
        print("x_two shape: ", str(x_two.shape))


        # combine all of our datas back together before returning it
        x_train = np.concatenate((x_one_train, x_two_train))
        y_train = np.concatenate((y_one_train, y_two_train))
        x_dev = np.concatenate((x_one_dev, x_two_dev))
        y_dev = np.concatenate((y_one_dev, y_two_dev))
        x_test = np.concatenate((x_one_test, x_two_test))
        y_test = np.concatenate((y_one_test, y_two_test))

        if verbose:
                print('train/dev/test ratios : ' +
                      str(ratio_train) + ', ' + str(ratio_dev) +
                      ', ' + str(ratio_test))
                print('x_train shape: ' + str(x_train.shape))
                print('y_train shape: ' + str(y_train.shape))
                print('x_dev shape: ' + str(x_dev.shape))
                print('y_dev shape: ' + str(y_dev.shape))
                print('x_test shape: ' + str(x_test.shape))
                print('y_test shape: ' + str(y_test.shape))

                
        return x_train, y_train, x_dev, y_dev, x_test, y_test




def save_kfold_accuracy(directory, model_name, scores, cm_string):
        from shutil import copyfile
        filename = str('kfold_' + str(scores.mean()))
        copyfile('./config.py', directory + '/' + model_name + '/' +
                 filename + ".txt")
        f = open(directory + '/' + model_name + '/' +  filename + '.txt', 'a+')
        f.write('\'\'\'')
        f.write("\n\nresults: \n\n")
        for i in range(0, len(cm_string)):
                f.write("\nkfold " + str(i+1) + '/' + str(len(cm_string)))
                f.write("\n" + cm_string[i])
        f.write("\nscores: " + str(scores))
        f.write("\nmean: " + str(scores.mean()))
        name = f.name
        f.write('\n\'\'\'')
        f.close()
        return name


def save_results(directory, model_name, history):
        '''
        copy the config.py file to directory/model_name/accuracy.txt
        append the results to the bottom of the file as a python block comment
        '''

        if not os.path.exists(directory +'/' + model_name):
                os.makedirs(directory + '/' + model_name)
                
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

        f.write('\'\'\'')

        f.close()

def initialize_output_directory(directory, model_name):
        if not os.path.exists(directory):
                os.makedirs(directory)
        if len(model_name) == 1:
                if not os.path.exists(directory + '/' + model_name[0]):
                        os.makedirs(directory + '/' + model_name[0])
        else:
                if not os.path.exists(directory + '/' + "ensemble"):
                        os.makedirs(directory + '/' + "ensemble")

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
        loaded_params['data_augmentation_directory'] = config.data_augmentation_directory
        loaded_params['k_folds'] = config.k_folds
        loaded_params['use_class_weights'] = config.use_class_weights
        loaded_params['use_oversampling'] = config.use_oversampling
        loaded_params['use_data_augmentation'] = config.use_data_augmentation
        loaded_params['use_attention_networks'] = config.use_attention_networks
        loaded_params['fine_tuning'] = config.fine_tuning
        gpus =  get_available_gpus()
        if gpus >= config.num_gpus:
                loaded_params['num_gpus'] = config.num_gpus
        else:
                loaded_params['num_gpus'] = gpus
                print('Number of specified gpus in config.py exceeds available gpus, using all available gpus')

                
        print("Using {", str(loaded_params['num_gpus']), "} gpus")

        if loaded_params["num_gpus"] == 1:
                print('Using gpu: ', str(config.gpu_to_use))
        loaded_params['gpu_to_use'] = config.gpu_to_use;
        
        return loaded_params

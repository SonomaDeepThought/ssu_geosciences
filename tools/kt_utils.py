import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # allow truncated images to load 

from keras.preprocessing.image import ImageDataGenerator

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
        for filename in os.listdir(folder):
                img = Image.open(os.path.join(folder, filename))
                if img is not None:
                        rbgimg = Image.new("RGB", img.size)
                        rbgimg.paste(img)
                        rbgimg = rbgimg.resize((img_size, img_size), Image.ANTIALIAS)
                        np_img = np.array(rbgimg)
                        images.append(np_img)

        return images


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

        batch_size = 32
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
        p = x.copy()
        q = y.copy()
        num_added = 0
        for x_batch, y_batch in datagen.flow(x, y, batch_size=batch_size,
                                             save_to_dir=directory,
                                             save_prefix='data_aug',
                                             save_format='jpeg'):
                p = np.concatenate((p, x_batch))
                q = np.concatenate((q, y_batch))
                num_added += x_batch.shape[0]
                if num_added >= num_data_to_add:
                        break
                        


        return p,q


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


def load_dataset(image_directory, img_size, ratio_train = 0.6, ratio_dev = -1,
                 ratio_test = -1, use_data_augmentation = False, use_oversampling=False,
                 verbose = False, data_augment_directory=None):

        '''
        Generate x_train, y_train, x_dev, y_dev, x_test, y_test from images in the
        provided image directory

        TODO - refactor into something more readable. 
        TODO - Allow this function to work with more than just a binary set of data
        '''
        
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

        if use_oversampling:
                use_data_augmetation = False
        
        delta_size_one = 0
        delta_size_two = 0

        num_x_one = len(next(os.walk(image_directory + '/'
                                     + dirs[0]))[2])
        num_x_two = len(next(os.walk(image_directory + '/' +
                                     dirs[1]))[2])                
                
        if use_data_augmentation and not use_oversampling:
                if not os.path.exists(data_augment_directory):
                        os.makedirs(data_augment_directory)
                        if not os.path.exists(data_augment_directory + '/' + dirs[0]):
                                os.makedirs(data_augment_directory + '/' + dirs[0])
                        if not os.path.exists(data_augment_directory + '/' + dirs[1]):
                                os.makedirs(data_augment_directory + '/' + dirs[1])

                
                num_x_aug_one = len(next(os.walk(data_augment_directory +
                                         '/' + dirs[0]))[2])
                num_x_aug_two = len(next(os.walk(data_augment_directory +
                                         '/' + dirs[1]))[2])

                
                num_x_one += num_x_aug_one
                num_x_two += num_x_aug_two
        

                delta_size_one = num_x_two - num_x_one
                delta_size_two = num_x_one - num_x_two



                
        x_one = np.array(load_images(image_directory + '/' + dirs[0],
                                     img_size))
        x_two = np.array(load_images(image_directory + '/' + dirs[1],
                                     img_size))


                        
                        
        y_one = np.ones((x_one.shape[0], 1))
        y_two = np.zeros((x_two.shape[0], 1))

        if use_oversampling and not use_data_augmentation:
                delta_size_one = num_x_two - num_x_one
                delta_size_Two = num_x_one - num_x_two

                # duplicate the larger delta into our image array
                x_one, y_one = oversample(x_one, y_one, delta_size_one)
                x_two, y_two = oversample(x_two, y_two, delta_size_two)


        if use_data_augmentation and not use_oversampling:
                print("Loading Augmented images...")
                x_one_aug = np.array(load_images(data_augment_directory +
                                                 '/' + dirs[0],
                                                 img_size))
                x_two_aug = np.array(load_images(data_augment_directory +
                                                 '/' + dirs[1],
                                                 img_size))
                if x_one_aug.shape[0] != 0:
                        print("Loaded ", str(x_one_aug.shape[0]), " augmented images")
                        x_one = np.concatenate((x_one, x_one_aug))
                if x_two_aug.shape[0] != 0:
                        print("Loaded ", str(x_two_aug.shape[0]), " augmented images")
                        x_two = np.concatenate((x_two, x_two_aug))

                        
                        # create and append delta_size to our array of images
                        x_one, y_one = data_augment(x_one, y_one,
                                                    delta_size_one,
                                                    data_augment_directory + '/' + dirs[0])
                        x_two, y_two = data_augment(x_two, y_two,
                                                    delta_size_two,
                                                    data_augment_directory + '/' + dirs[1])

        
        # X_images is the concatenation of all images
        # Y_images is the concatenation of all labels
        Y_images = np.concatenate((y_one, y_two))
        X_images = np.concatenate((x_one, x_two))
        
        # setup the training set
        x_train_all = load_subset(x_one, x_two, ratio_train, 0, 0, 0, 0)
        y_train_all = load_subset(y_one, y_two, ratio_train, 0, 0, 0, 0)
        
        # setup the dev set
        x_dev_all = load_subset(x_one, x_two, ratio_dev,
                                x_train_all['num_one'], x_train_all['num_two'],
                                x_train_all['num_one'], x_train_all['num_two'])

        y_dev_all = load_subset(y_one, y_two, ratio_dev,
                                y_train_all['num_one'], y_train_all['num_two'],
                                y_train_all['num_one'], y_train_all['num_two'])

        # setup the test set
        x_test_all = load_subset(x_one, x_two, ratio_test,
                                 x_train_all['num_one']+x_dev_all['num_one'],
                                 x_train_all['num_two'] + x_dev_all['num_two'],
                                 x_one.shape[0], x_two.shape[0])

        y_test_all = load_subset(y_one, y_two, ratio_test,
                                 y_train_all['num_one']+y_dev_all['num_one'],
                                 y_train_all['num_two'] + y_dev_all['num_two'],
                                 y_one.shape[0], y_two.shape[0])

        # retrieve results
        x_train = x_train_all['results']
        x_dev = x_dev_all['results']
        y_train = y_train_all['results']
        y_dev = y_dev_all['results']
        x_test = x_test_all['results']
        y_test = y_test_all['results']


        if verbose:
                print('train/dev/test ratios : ' +
                      str(ratio_train) + ', ' + str(ratio_dev) +
                      ', ' + str(ratio_test))
                print('x_images shape: ' + str(X_images.shape))
                print('y_images shape: ' + str(Y_images.shape))
                print('x_train shape: ' + str(x_train.shape))
                print('y_train shape: ' + str(y_train.shape))
                print('x_dev shape: ' + str(x_dev.shape))
                print('y_dev shape: ' + str(y_dev.shape))
                print('x_test shape: ' + str(x_test.shape))
                print('y_test shape: ' + str(y_test.shape))

        # check to ensure our sizes match
        assert(X_images.shape[0] == (x_train.shape[0] +
                                     x_dev.shape[0] +
                                     x_test.shape[0]))

        assert(Y_images.shape[0] == (y_train.shape[0] +
                                     y_dev.shape[0] +
                                     y_test.shape[0]))

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
        loaded_params['data_augmentation_directory'] = config.data_augmentation_directory
        loaded_params['k_folds'] = config.k_folds
        loaded_params['use_class_weights'] = config.use_class_weights
        loaded_params['use_oversampling'] = config.use_oversampling
        loaded_params['use_data_augmentation'] = config.use_data_augmentation
        loaded_params['use_attention_networks'] = config.use_attention_networks
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

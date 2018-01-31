# surpress tensorflow warnings
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # log minimum errors to the user

import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import StratifiedKFold

import time

# import our local files
from kt_utils import *
from model import *

def train_and_evaluate_model(model, X_train, Y_train, X_dev, Y_dev,
                             batch_size=32, num_epochs=1):
    '''
    Inputs: 
        model: Keras model to train on
        X_train: the images to train on in the form (#images, height, width, channels)
        Y_train: the labels for the X_train in the form (#images, label)

        X_dev: the images to test on in the same form as X_train

        Y_dev: the labels for the X_dev in the same form as Y_train

        batch_size: number of images per minibatch
    
        num_epochs: number of epochs to train for

    Returns:
        History. A keras History object with history.history being a dictionary
        of ['acc'] ['loss'] ['val_acc']  ['val_loss'] , each of which are lists

        e.g.  history.history['acc'][0]  could produce a float with the results        for the first epoch's accuracy
    '''

    train_datagen = ImageDataGenerator(
        rotation_range=40, # degrees we can rotate max 180
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


    test_datagen = ImageDataGenerator()
    
    train_datagen.fit(X_train)
    test_datagen.fit(X_dev)

    '''
    history =  model.fit_generator(train_datagen.flow(X_train,
                                                      Y_train,
                                                      batch_size=batch_size),
                                   samples_per_epoch=X_train.shape[0] / batch_size,
                                   epochs = num_epochs,
                                   validation_data=test_datagen.flow(X_dev,
                                                                     Y_dev,
                                                                     batch_size=batch_size),
                                   nb_val_samples=X_dev.shape[0] / batch_size)


    '''

    history =  model.fit_generator(train_datagen.flow(X_train,
                                                      Y_train,
                                                      batch_size=batch_size),
                                   epochs = num_epochs)


    # add confusion matrix prediction code.
    # preds = model.predict(X_dev, Y_dev)
    return history

def k_fold(model, X_train, Y_train, X_dev, Y_dev, batch_size, num_epochs):

    # no imagedatagen being used in kfold yet.
    
    model.fit(X_train, Y_train, epochs=num_epochs,
              batch_size=batch_size)
    results = model.evaluate(X_dev, Y_dev)
    preds = model.predict(X_dev)
    return preds, results[1] # return our preds, accuracy

def main(loaded_params):


    model_name = loaded_params['model_name']
    num_epochs = loaded_params['num_epochs']
    batch_size = loaded_params['batch_size']
    ratio_train = loaded_params['ratio_train']
    ratio_test = loaded_params['ratio_test']
    learning_rate = loaded_params['learning_rate']
    output_directory = loaded_params['output_directory']
    optimizer = loaded_params['optimizer']
    image_directory = loaded_params['image_directory']
    num_gpus = loaded_params['num_gpus']
    k_folds = loaded_params['k_folds']
    
    base_model, img_size = load_base_model(model_name)

    # load our images
    X_train_orig, Y_train_orig, X_dev_orig, Y_dev_orig, X_test_orig, Y_test_orig  = load_dataset(image_directory, img_size, ratio_train=ratio_train, ratio_test = ratio_test)

    # Normalize image vectors
    X_train = X_train_orig/255.
    X_dev = X_dev_orig/255.
    X_test = X_test_orig/255.

    # Rename
    Y_train = Y_train_orig
    Y_dev = Y_dev_orig
    Y_test = Y_test_orig

    
    print_shapes(X_train, Y_train, X_dev, Y_dev, X_test, Y_test)
    
        
    completed_model = create_final_layers(base_model,
                                          img_size,
                                          learning_rate=learning_rate,
                                          optimizer=optimizer, num_gpus=num_gpus)

    completed_model.summary() # print to the user the summary of our model

    # for k-fold we must combine our data into a single entity.
    data = np.concatenate((X_train, X_dev), axis=0)
    labels = np.concatenate((Y_train, Y_dev), axis=0)
    skf = StratifiedKFold(n_splits = k_folds)
    skf.get_n_splits(data, labels)
#    skf = StratifiedKFold(labels[:,0], n_folds=k_folds, shuffle=True)

    
    if k_folds <= 1:
        history = train_and_evaluate_model(completed_model,
                                           X_train,
                                           Y_train,
                                           X_dev,
                                           Y_dev,
                                           batch_size=batch_size,
                                           num_epochs=num_epochs)

    else:

        scores = np.zeros(k_folds)
        idx = 0
        
        for (train, test) in skf.split(data,labels):
            #print ("Running Fold", i+1, "/", k_folds)
            completed_model = None
            completed_model = create_final_layers(base_model,
                                                  img_size,
                                                  learning_rate=learning_rate,
                                                  optimizer=optimizer,
                                                  num_gpus=num_gpus)

            preds, scores[idx] = k_fold(completed_model, data[train], labels[train],
                                 data[test], labels[test],
                                 batch_size=batch_size, num_epochs=num_epochs)
            idx += 1
            cm = confusion_matrix(labels[test],
                                           preds, labels=[0,1])
                                           
            print_cm(cm, labels=['Negative', 'Positive'])
            
    print("\nscores: ", str(scores))
    print("mean: ", str(scores.mean()))
#    save_results(output_directory, model_name, history)
   
    # stop the session from randomly failing to exit gracefully
    K.clear_session() 



if __name__ == "__main__":

    start = time.time()
    loaded_params = parse_config_file()
    initialize_output_directory(loaded_params['output_directory'],
                                loaded_params['model_name'])
    main(loaded_params)
    end = time.time()
    print("total elapsed time: ", str(end -start))

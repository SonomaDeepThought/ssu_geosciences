# surpress tensorflow warnings
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # log minimum errors to the user

# import our passed number of gpus by user
from config import gpu_to_use as gtu
from config import num_gpus as ngpu

if ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gtu - 1)
elif ngpu == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
import tensorflow as tf

from tools.kt_utils import *
from tools.training import *
from model import *

import numpy as np
np.random.seed(1337)
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight



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
    data_augmentation_directory = loaded_params['data_augmentation_directory']
    num_gpus = loaded_params['num_gpus']
    k_folds = loaded_params['k_folds']
    use_class_weights = loaded_params['use_class_weights']
    use_oversampling = loaded_params['use_oversampling']
    use_data_augmentation = loaded_params['use_data_augmentation']
    use_attention_networks = loaded_params['use_attention_networks']
    fine_tuning = loaded_params['fine_tuning']

    base_models = []
    if len(model_name) > 1:
        input_shape = (224,224,3)
    else:
        input_shape = None
    
    for model in model_name:
        base_model, img_size = load_base_model(model, fine_tuning=fine_tuning,
                                               input_shape=input_shape)
        base_models.append(base_model)

    # load our images
    X_train_orig, Y_train_orig, X_dev_orig, Y_dev_orig, X_test_orig, Y_test_orig  = load_dataset(image_directory, img_size, ratio_train=ratio_train, ratio_test = ratio_test, use_data_augmentation=use_data_augmentation, data_augment_directory=data_augmentation_directory, use_oversampling=use_oversampling)

    # Normalize image vectors
    X_train = X_train_orig/255.
    X_dev = X_dev_orig/255.
    X_test = X_test_orig/255.

    # Rename
    Y_train = Y_train_orig
    Y_dev = Y_dev_orig
    Y_test = Y_test_orig

    
    print_shapes(X_train, Y_train, X_dev, Y_dev, X_test, Y_test)

    
    if k_folds == None or k_folds <= 1:
        print("building models")
        if len(model_name) > 1:
            output_directory += '/ensemble'
        model_preds = []
        i = 0
        for model in base_models:
            completed_model = create_final_layers(model,
                                                  img_size,
                                                  learning_rate=learning_rate,
                                                  optimizer=optimizer,
                                                  num_gpus=num_gpus)
            
            print('finished building model\nTraining Model')
            history, preds = train_and_evaluate_model(completed_model,
                                               X_train,
                                               Y_train,
                                               X_dev,
                                               Y_dev,
                                               batch_size=batch_size,
                                               num_epochs=num_epochs,
                                               use_class_weights=use_class_weights)
            cm = confusion_matrix(Y_dev,
                                  preds, labels=[0,1])

            print_cm(cm, labels=['Non-Sigma', 'Sigma'])
            model_preds.append(preds)
            save_results(output_directory, model_name[i], history)
            i += 1


        # handle ensemble operations
        avg_preds = np.average(model_preds, axis=0)
        # store our images
        save_images(avg_preds, Y_dev, X_dev, "ensemble",  ensembles=True)
        avg_preds = avg_preds > 0.5 # apply binary classifier thresholding        
        avg_correct = Y_dev == avg_preds # get an array of correct answers
        print("----------------------------------------")
        print("ensemble accuracy: ",
              str(np.sum(avg_correct) / len(avg_correct)))

        cm = confusion_matrix(Y_dev,
                              avg_preds, labels=[0,1])

        print_cm(cm, labels=['Non-Sigma', 'Sigma'])
        
        
    else:

        # for k-fold we must combine our data into a single entity.
        data = np.concatenate((X_train, X_dev), axis=0)
        labels = np.concatenate((Y_train, Y_dev), axis=0)

        # we shuffle to ensure that on any given kfold we are not
        # simply training/testing on a single class 
        skf = StratifiedKFold(n_splits = k_folds, shuffle=True)
        scores = np.zeros(k_folds)
        idx = 0
        cm_strings = []
        for (train, test) in skf.split(data,labels):
            print ("Running Fold", idx+1, "/", k_folds)
            base_model = None
            base_model, img_size = load_base_model(model_name, fine_tuning=fine_tuning)
        
            completed_model = None

            completed_model = create_final_layers(base_model,
                                                  img_size,
                                                  learning_rate=learning_rate,
                                                  optimizer=optimizer,
                                                  num_gpus=num_gpus)
            start = time.time()
            preds, scores[idx] = k_fold(completed_model,
                                       data[train], labels[train],
                                        data[test], labels[test],
                                        batch_size=batch_size,
                                        num_epochs=num_epochs,
                                        use_class_weights=use_class_weights)

            print("time to k_fold: ", str(time.time()-start))
            idx += 1
            cm = confusion_matrix(labels[test],
                                           preds, labels=[0,1])
                                           
            cm_strings.append(print_cm(cm, labels=['Negative', 'Positive']))
            
        print("\nscores: ", str(scores))
        print("mean: ", str(scores.mean()))
        save_kfold_accuracy(output_directory, model_name, scores, cm_strings)
        






    
if __name__ == "__main__":
    
    import time
    start = time.time()
    loaded_params = parse_config_file()
    initialize_output_directory(loaded_params['output_directory'],
                                loaded_params['model_name'])


    main(loaded_params)
    end = time.time()
    print("total elapsed time: ", str(end -start))

    # stop the session from ending after main has finished, exit gracefully
    K.clear_session() 




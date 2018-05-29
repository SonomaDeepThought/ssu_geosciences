# surpress tensorflow warnings
from sklearn import svm
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
np.random.seed(1337) # limit the randomness between sessions.
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
#from tqdm import tqdm


SVM_Feature_Extractor = False

def main(loaded_params):
    '''
    input : a dictionary of parameters loaded from the config.py
    '''
    
    
    model_name = loaded_params['model_name']
    ratio_train = loaded_params['ratio_train']
    ratio_test = loaded_params['ratio_test']
    output_directory = loaded_params['output_directory']
    k_folds = loaded_params['k_folds']

    use_attention_networks = loaded_params['use_attention_networks'] # not currently used

    
    base_models = []

    # if our model is an ensemble, default to 224,224,3 
    if len(model_name) > 1:
        input_shape = (224,224,3)
    else:
        input_shape = None


    # load all of the models into base_models
    for model in model_name:
        base_model, img_size = load_base_model(model,
                                               fine_tuning=loaded_params['fine_tuning'],
                                               input_shape=input_shape)
        base_models.append(base_model)


        
    # load our images
    X_train_orig, Y_train_orig, X_dev_orig, Y_dev_orig, X_test_orig, Y_test_orig, labels  = load_dataset(loaded_params['image_directory'], img_size, ratio_train=ratio_train, ratio_test = ratio_test, use_data_augmentation=loaded_params['use_data_augmentation'], data_augment_directory=loaded_params['data_augmentation_directory'], use_oversampling=loaded_params['use_oversampling'])


    print(labels)

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
        # handle no k-fold

        
        print("building models")
        if len(model_name) > 1:
            output_directory += '/ensemble'
        model_preds = []
        i = 0
        for model in base_models:
            completed_model = create_final_layers(model,
                                                  img_size,
                                                  labels=labels,
                                                  learning_rate=loaded_params['learning_rate'],
                                                  optimizer=loaded_params['optimizer'],
                                                  num_gpus=loaded_params['num_gpus'])
            
            print('finished building model\nTraining Model')
            history, preds = train_and_evaluate_model(completed_model,
                                               X_train,
                                               Y_train,
                                               X_dev,
                                               Y_dev,
                                               batch_size=loaded_params['batch_size'],
                                               num_epochs=loaded_params['num_epochs'],
                                               use_class_weights=loaded_params['use_class_weights'])
            print('preds.shape: ' + str(preds.shape))
            print('Y_dev.shape: ' + str(Y_dev.shape))

            cm = confusion_matrix(Y_dev,
                                  preds, labels=[0,1])


            print_cm(cm, labels=['Non-Sigma', 'Sigma'])
            model_preds.append(preds)
            save_results(output_directory, model_name[i], history)
            i += 1

            print(str(completed_model.predict(np.expand_dims(X_dev[55], axis=0))))
        # handle ensemble operations
        avg_preds = np.average(model_preds, axis=0)
        # store our images
        save_images(avg_preds, Y_dev, X_dev, "ensemble",  ensembles=True)
        #avg_preds = avg_preds > 0.5 # apply binary classifier thresholding
        avg_preds = (avg_preds == avg_preds.max(axis=1)[:,None]).astype(int)
        print(avg_preds)

        # get number of correct
        avg_correct = np.sum(np.all(avg_preds == Y_dev, axis=1))
        print("----------------------------------------")
        print("ensemble accuracy: ",
              str(np.sum(avg_correct) / len(Y_dev)))

        cm = confusion_matrix(Y_dev,
                              avg_preds, labels=[0,1])

        print_cm(cm, labels=['Non-Sigma', 'Sigma'])
        

        
        
    else:
        # using k-fold cross validation

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
            base_model, img_size = load_base_model(model_name,
                                                   fine_tuning=loaded_params['fine_tuning'])
        
            completed_model = None

            completed_model = create_final_layers(base_model,
                                                  img_size,
                                                  learning_rate=loaded_params['learning_rate'],
                                                  optimizer=loaded_params['optimizer'],
                                                  num_gpus=loaded_params['num_gpus'])
            start = time.time()
            preds, scores[idx] = k_fold(completed_model,
                                       data[train], labels[train],
                                        data[test], labels[test],
                                        batch_size=loaded_params['batch_size'],
                                        num_epochs=loaded_params['num_epochs'],
                                        use_class_weights=loaded_params['use_class_weights'])

            print("time to k_fold: ", str(time.time()-start))
            idx += 1
            cm = confusion_matrix(labels[test],
                                           preds, labels=[0,1])
                                           
            cm_strings.append(print_cm(cm, labels=['Negative', 'Positive']))
            
        print("\nscores: ", str(scores))
        print("mean: ", str(scores.mean()))
        save_kfold_accuracy(output_directory, model_name, scores, cm_strings)
        
    if SVM_Feature_Extractor:
        print("Now running feature extractor...")
        all_outs=[]
        print("Extracting features from test set...")
        for i in tqdm(range(len(X_train))):
            pic = np.multiply(np.ones((1,227,227,3)),X_train[i])
            intermediate_layer_model = Model(inputs=completed_model.input, outputs=completed_model.get_layer(index=-3).output)
            intermediate_output = np.squeeze(intermediate_layer_model.predict(pic), axis=0)
            all_outs.append(intermediate_output)
        print("Output Shape: ", end="")
        print(np.shape(all_outs))
        print("creating SVM...")
        clf = svm.SVC()
        print("Fitting SVM...")
        clf.fit(all_outs, Y_train)
    
        svm_preds = []
        print("Extracting features from dev set and making predictions using SVM...")
        for i in tqdm(range(len(X_dev))):
            pic = np.multiply(np.ones((1,227,227,3)),X_dev[i])
            intermediate_layer_model = Model(inputs=completed_model.input, outputs=completed_model.get_layer(index=-3).output)
            intermediate_output = np.squeeze(intermediate_layer_model.predict(pic), axis=0)
            svm_preds.append(clf.predict([intermediate_output]))
        total_accuracy = sum(svm_preds==Y_dev)/len(Y_dev)
        print("Total accuracy of the SVM as a feature extractor:", total_accuracy)





    
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




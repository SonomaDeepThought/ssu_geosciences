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
np.random.seed(1337)
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.utils import class_weight
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib


SVM_Feature_Extractor = True
SVM_find_optimum = False

######### SVM ##########
def svc_param_selection(X, y):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=cv)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

######## END SVM ########

def CNN_features_extracted(X_train, model):
        print("Now running feature extractor...")
        all_outs=[]
        print("Extracting features from test set...")
        for i in tqdm(range(len(X_train))):
            pic = np.multiply(np.ones((1,227,227,3)),X_train[i])
            intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(index=-3).output)
            intermediate_output = np.squeeze(intermediate_layer_model.predict(pic), axis=0)
            all_outs.append(intermediate_output)
        print("Output Shape: ", end="")
        print(np.shape(all_outs))
        return all_outs

def SVM_params(svm_in, Y_train):
        if SVM_find_optimum:
            best_params = svc_param_selection(svm_in, Y_train)
            print("best_params are: ", best_params)
        else:
            best_params = {'C':10, 'gamma':0.001}
        return best_params

def create_SVM(best_params, svm_in, Y_train):
        C = best_params['C']
        gamma = best_params['gamma']
        print("creating SVM...")
        clf = svm.SVC(C=C, gamma=gamma)
        print("Fitting SVM...")
        clf.fit(svm_in, Y_train)
        return clf

def test_svm(clf, model, X_dev, Y_dev):
        svm_preds = []
        print("Extracting features from dev set and making predictions using SVM...")
        for i in tqdm(range(len(X_dev))):
            pic = np.multiply(np.ones((1,227,227,3)),X_dev[i])
            intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(index=-3).output)
            intermediate_output = np.squeeze(intermediate_layer_model.predict(pic), axis=0)
            svm_preds.append(clf.predict([intermediate_output]))
        total_accuracy = sum(svm_preds==Y_dev)/len(Y_dev)
        save_images(svm_preds, Y_dev, X_dev, "svm", ensembles=False) 
        print("Total accuracy of the SVM as a feature extractor:", total_accuracy)
        #cm = confusion_matrix(Y_dev, svm_preds, labels=[0,1])

        #print_cm(cm, labels=['Non-Sigma', 'Sigma'])


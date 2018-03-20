from sklearn.utils import class_weight
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tools.kt_utils import *

def train_and_evaluate_model(model, X_train, Y_train, X_dev, Y_dev,
                             batch_size=32, num_epochs=1,
                             use_class_weights=False):
    '''                                                                         
    Inputs:                                                                     
    model: Keras model to train on                                          
    X_train: the images to train on in the form (#images, height, width, ch\
    annels)                                                                         
    Y_train: the labels for the X_train in the form (#images, label)        
    
    X_dev: the images to test on in the same form as X_train                
    
    Y_dev: the labels for the X_dev in the same form as Y_train             
    
    batch_size: number of images per minibatch                              
    
    num_epochs: number of epochs to train for                               
    
    Returns:                                                                  
    
    History. A keras History object with history.history being a dictionary
    of ['acc'] ['loss'] ['val_acc']  ['val_loss'] , each of which are lists
    
    e.g.  history.history['acc'][0]  could produce a float with the results
    for the first epoch's accuracy                                          
    '''


    

    cw = None
    if use_class_weights:
        cw = get_class_weights(Y_train)
    print('class weights: ', str(cw))
    
    history = model.fit(X_train,
                        Y_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        class_weight=cw)
                        
    
    print("eval: ", str(model.evaluate(X_dev, Y_dev)))
    preds = model.predict(X_dev)



    save_images(preds, Y_dev, X_dev, model.layers[1].name)


    return history, preds


def k_fold(model, X_train, Y_train, X_dev, Y_dev, batch_size, num_epochs, use_class_weights=True):

    # no imagedatagen being used in kfold yet.
    print("X_train shape: ", str(X_train.shape))


    
    cw = None
    if use_class_weights:
        cw = get_class_weights(Y_train)
    print('class weights: ', str(cw))
    model.fit(X_train, Y_train, epochs=num_epochs,
              batch_size=batch_size, class_weight=cw)
    results = model.evaluate(X_dev, Y_dev)
    preds = model.predict(X_dev)
    print('k_fold accuracy: ', str(results[1]))
    return preds, results[1] # return our preds, accuracy


def get_class_weights(Y_train):
    cw = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                          np.unique(Y_train),
                                                          Y_train[:,0])))
    return cw




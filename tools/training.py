from sklearn.utils import class_weight
import numpy as np

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
    cw = None
    if use_class_weights:
        cw = get_class_weights(Y_train)
    print('cw: ', str(cw))
    history =  model.fit_generator(train_datagen.flow(X_train,
                                                      Y_train,
                                                      batch_size=batch_size),
                                   epochs = num_epochs,
                                   class_weight=cw)
    
    
    
    # preds = model.predict(X_dev, Y_dev)
    return history


def k_fold(model, X_train, Y_train, X_dev, Y_dev, batch_size, num_epochs, use_class_weights=True):

    # no imagedatagen being used in kfold yet.
    print("X_train shape: ", str(X_train.shape))


    
    cw = None
    if use_class_weights:
        cw = get_class_weights(Y_train)
    print('cw: ', str(cw))
    model.fit(X_train, Y_train, epochs=num_epochs,
              batch_size=batch_size, class_weight=cw)
    results = model.evaluate(X_dev, Y_dev)
    preds = model.predict(X_dev)
    return preds, results[1] # return our preds, accuracy


def get_class_weights(Y_train):
    cw = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                          np.unique(Y_train),
                                                          Y_train[:,0])))
    return cw




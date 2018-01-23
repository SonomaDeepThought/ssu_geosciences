import keras 

from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Model


def create_final_layers(base_model, img_size, optimizer=None,
                        learning_rate=0.001, dropout_rate=0.5):

    """
    Inputs:
       base_model: Keras.application, this is the CNN 
                   that we will be adding too. 
       img_size: height and width to rescale our images to

       optimizer: string name of the keras optimizer to use
    
       learning_rate: float that determines how fast our model learns

       dropout_rate: the percent of nodes in our dropout layer to drop

    Returns:
        returns the completed model that we should then train on
    """
    input_shape=(img_size, img_size, 3)
    input = Input(shape=input_shape, name = 'image_input')
    output_conv = base_model(input)

    x = Dense(4096, activation='relu',
              name = 'fc_dense1')(output_conv)
    x = Dropout(dropout_rate, name='fc_dropout1')(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)

    model = Model(input=input, output=x)

    if optimizer is None:
        optimizer = keras.optimizers.Adam(lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(lr=learning_rate)
    elif optimizer == 'adam':
        optimizer = keras.optimizers.Adam(lr=learning_rate)
    elif optimizer == 'RMSProp':
        optimizer = keras.optimizers.RMSprop(lr=learning_rate)
    elif optimizer == 'adagrad':
        optimizer = keras.optimizers.Adagrad(lr=learning_rate)
    elif optimizer == 'adadelta':
        optimizer = keras.optimizers.Adadelta(lr=learning_rate)
    elif optimizer == 'adamax':
        optimizer = keras.optimizers.Adamax(lr=learning_rate)
    elif optimizer == 'nadam':
        optimizer = keras.optimizers.Nadam(lr=learning_rate)
    else:
        print("optimizer name not recognized")

    print("optimizer: ", optimizer)
    print("optimizer config: ", optimizer.get_config())
        
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model





def load_base_model(model_name, input_shape=None):
    '''
    model_name : The name of the model we wish to implement
    input_shape : the tensor shape of the image we wish to use on our model

    returns
    base_model : The model created based on our model_name 
    img_size : the height and width of our image used in our current model
    '''

    if input_shape is not None:
        img_size = input_shape[0]
        
    if model_name == 'InceptionV3':
        if input_shape == None:
            img_size = 227
            input_shape = (img_size, img_size, 3)
        base_model = keras.applications.inception_v3.InceptionV3(weights='imagenet',
                                 include_top = False,
                                 input_shape=input_shape,
                                 pooling = 'avg')


    elif model_name == 'ResNet50':
        if input_shape == None:
            img_size = 224
            input_shape = (img_size, img_size, 3)
        base_model = keras.applications.resnet50.ResNet50(weights='imagenet',
                              include_top = False,
                              input_shape = input_shape,
                              pooling = 'avg')


    elif model_name == 'VGG16':
        if input_shape == None:
            img_size = 224
            input_shape = (img_size, img_size, 3)
        base_model = keras.applications.vgg16.VGG16(weights='imagenet',
                           include_top = False,
                           input_shape = input_shape,
                           pooling = 'avg')


    elif model_name == 'VGG19':
        if input_shape == None:
            img_size = 224
            input_shape = (img_size, img_size, 3)
        base_model = keras.applications.vgg19.VGG19(weights='imagenet',
                           include_top=False,
                           input_shape=input_shape,
                           pooling='avg')


    elif model_name == 'InceptionResNetV2':
        if input_shape == None:
            img_size = 299
            input_shape = (img_size, img_size, 3)
        base_model = keras.applications.inception_resnet_v2.InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            pooling='avg')


    elif model_name == 'Xception':
        if input_shape == None:
            img_size = 299
            input_shape=(img_size, img_size, 3)
        base_model = keras.applications.xception.Xception(weights='imagenet',
                              include_top = False,
                              input_shape=input_shape,
                              pooling = 'avg')


    else:
        print("model name not recognized.")
        return

    print('\n' + base_model.name, ' base model with input shape',
          base_model.input_shape, ' loaded')

    # Since we are implementing transfer learning, we do not wish to train the
    # base model
    for layer in base_model.layers:
        layer.trainable = False

    return base_model, img_size
            
            
            
            
        



def sigmaclast_model(input_shape):
    """                                                                        
    Implementation of the HappyModel. Taken from coursera HappyModel lab      

    Arguments:                                                                 
    input_shape -- shape of the images of the dataset                          

    Returns:                                                                   
    model -- a Model() instance in Keras                                       
    """
    
    
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)
    
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    
    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)
    
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='SigmaClastModel')
    
    
    return model





def identity_block(X, f, filters, stage, block):
    """                                                                        
    Implementation of the identity block as defined in Figure 3               
                                                                             
    Arguments:                                                                 
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)               
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path                                                     
    stage -- integer, used to name the layers, depending on their position in the network     
    block -- string/character, used to name the layers, depending on their position in the network                                                             
                                                                                
    Returns:                                                                    
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)          
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third componoent of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3 , name = bn_name_base + '2c')(X)
    
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = layers,add([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    """                                                                         
    Implementation of the convolutional block as defined in Figure 4            
                                                                                
    Arguments:                                                                  
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)                
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path                                                     
    stage -- integer, used to name the layers, depending on their position in the network 
    block -- string/character, used to name the layers, depending on their position in the network                                                             
    s -- Integer, specifying the stride to be used                              
                                                                                
    Returns:                                                                    
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)     
    """

    # define name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    #### Main Path ####
    # First component in main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (s,s),
               name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same',
               name = conv_base_name + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third componoent of main path
    X = Conv2D(filters = F3, kernel_size = (1,1), strides =(1,1),
               name = conv_base_name + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #####
    X_shortcut = Conv2D(filters=F3, kernel_size = (1,1), strides =(s,s),
                        name=conv_name_base + '1',
                        kernel_initliazer=glorot_uniform(seed=0))(X)

    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path and pass it through relu actiation
    X = layers.add(inputs = [X, X_shortcut])
    X = Activaion('relu')(X)

    return X




def coursera_ResNet50(input_shape = (64, 64, 3), classes = 6):
        """                                                                         
    Implementation of the popular ResNet50 the following architecture:          
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLO\
CK -> IDBLOCK*3                                                                 
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER  
                                                                                
    Arguments:                                                                  
    input_shape -- shape of the images of the dataset                           
    classes -- integer, number of classes                                       
                                                                                
    Returns:                                                                    
    model -- a Model() instance in Keras                                        
    """



    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)


    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer\
               = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)


    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block\
                            ='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block\
                            ='a', s = 2)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b')
    X = identity_block(X, 3, [128,128,512], stage=3, block='c')
    X = identity_block(X, 3, [128,128,512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, bl\
                            ock='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    
    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters =  [512, 512, 2048], stage = 5, b\
                            lock='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2,2))(X)


    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_i\
              nitializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    
    return model
                                                        


    

from tools.kt_utils import *
import numpy as np
from PIL import Image

def test_oversampling():
    p = np.random.rand(200, 299, 299, 3)
    p_y = np.zeros((200,1))
    q = np.random.rand(100, 299, 299, 3)
    q_y = np.ones((100,1))

    delta_one = q.shape[0] - p.shape[0] # 100 - 200 == -100
    delta_two = p.shape[0] - q.shape[0] # 200 - 100 == 100
    
    print("p shape: ", str(p.shape))
    print("p_y shape: ", str(p_y.shape))
    print("q shape: ", str(q.shape))
    print("q_Y shape: ", str(q_y.shape))

    print("Oversampling")
    p, p_y = oversample(p, p_y, delta_one)
    q, q_y = oversample(q, q_y, delta_two)
    
    print("p shape: ", str(p.shape))
    print("p_y shape: ", str(p_y.shape))
    print("q shape: ", str(q.shape))
    print("q_Y shape: ", str(q_y.shape))
    
    
    return


def load_images(folder, img_size):
    '''                                                                     
    return an array containing all the images from the given folder.        
    all images are converted to RGB in channel_last format, and resized     
    to img_size x img_size                                                  
    '''
    import os
    
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            rbgimg = Image.new("RGB", img.size)
            rbgimg.paste(img)
            rbgimg = rbgimg.resize((img_size, img_size),
                                   Image.ANTIALIAS)
            np_img = np.array(rbgimg)
            images.append(np_img)
            
            
    return images
    

def test_load_images_not_random(image_folder, img_size):
    
    test_base = load_images(image_folder, img_size)
    results = True
    for i in range(0,20):
        test_2 = load_images(image_folder,img_size)
        results &= np.array_equal(test_base, test_2)

    print("test_load_images_not_random(): ", str(results))
    

#test_oversampling()
loaded_params = parse_config_file()

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


#base_model, img_size = load_base_model(model_name)
img_size = 299

test_load_images_not_random(image_directory + '/with', img_size)

'''
# load our images
X_train_orig, Y_train_orig, X_dev_orig, Y_dev_orig, X_test_orig, Y_test_orig  = load_dataset(image_directory, img_size, ratio_train=ratio_train, ratio_test = ratio_test, use_data_augmentation=use_data_augmentation, data_augment_directory=data_augmentation_directory, use_oversampling=use_oversampling)
'''


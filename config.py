# CNN model options:
# 'InceptionResNetV2'<-- 
# 'InceptionV3'
# 'ResNet50'
# 'VGG16'
# 'VGG19'
# 'Xception'
# 'SSUGeosciences'
<<<<<<< HEAD
model_name = [ 'InceptionV3', 'InceptionResNetV2', 'VGG16', 'VGG19']

batch_size = 32

num_epochs = 10
=======
model_name = ['InceptionV3']

batch_size = 32

num_epochs = 2
>>>>>>> edbd7d3cd5e8f33734245d2d2d5b131e6f9a4b87

learning_rate = 0.0001

# when k_folds > 1 the ratio_learn and ratio_test are ignored.
# all folds have size (samples / k_folds)
k_folds = 1


# number of gpus to use.
num_gpus = 1

# if num_gpus is set to 1, use this to define the gpu to run on.
# if gpu_to_use does not exist in the list of gpus, the model will be run on
# the cpu. GPU numbers are [1,2,...n] for all n-gpus in the machine
gpu_to_use = 2


# class weights give weighting based on representation of the dataset.
# e.g. if you have 2 sets of data {x1,x2} and x1 has 2x the data of x2
# x2 will be given a weight of 2 while x1 maintains a weight of 1
use_class_weights = False


# - WARNING - this boolean clobbers use_data_augmentation
#
# Take the minority class and duplicate its images until it is within a small margin
# of the majority class.
use_oversampling = False


# to modify the data augmentation settings edit tools/kt_utils.py
# In data_augment() modify the ImageDataGenerator function params
use_data_augmentation = False


# -NOT YET IMPLEMENTED-
use_attention_networks = False

# enable training on the CNN for specific layers.
# A CNN has n-layers, we will enable training on layers n-fine_tuning sub i
# for all i in len(fine_tuning)
# that is all of these layers starting from the output of the CNN will be
# trained
<<<<<<< HEAD
fine_tuning = {1,2,3,4,5,6,7,8,8,10,11,12,13,14,15}
=======
fine_tuning = {}
>>>>>>> edbd7d3cd5e8f33734245d2d2d5b131e6f9a4b87

# % of images to use in the training set. The number of images used for the
# dev set are derived from train set and test set.
ratio_train = 0.7

# -NOT YET IN USE- % of images to use in the test set (note that test set is different from validation/dev set).
ratio_test = 0

# since this is a binary classifier, there must be 2 folders inside the
# image directory. Those folders must be named: with, without
image_directory = './images'

# results will be placed in the output directory. seperated by the
# model_name provided and stored as the final accuracy % on the training set.
output_directory = './results'

# Images created through data_augmentation will be placed in this folder
data_augmentation_directory = './data_augmentation'

# optimizer options:
# 'sgd'
# 'adam'
# 'RMSProp'
# 'adagrad'
# 'adadelta'
# 'adamax'
# 'nadam' 
# or None which defaults to 'adam'
# The optimizers will use default parameters except learning_rate which
# is stored above
optimizer = None

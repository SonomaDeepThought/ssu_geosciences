# CNN model options:
# 'InceptionResNetV2'<-- 
# 'InceptionV3'
# 'ResNet50'
# 'VGG16'
# 'VGG19'
# 'Xception'
# 'SSUGeosciences'
model_name = 'InceptionResNetV2'

batch_size = 32

num_epochs = 2

learning_rate = 0.001

# when k_folds > 1 the ratio_learn and ratio_test are ignored.
# all folds have size (samples / k_folds)
k_folds = 1

use_class_weights = True

# % of images to use in the training set. The number of images used for the
# dev set are derived from train set and test set.
ratio_train = 0.7

# % of images to use in the test set (note that test set is different from validation/dev set).
ratio_test = 0

# number of gpus to use.
num_gpus = 2

# since this is a binary classifier, there must be 2 folders inside the
# image directory. Those folders must be named: with, without
image_directory = './images'

# results will be placed in the output directory. seperated by the
# model_name provided and stored as the final accuracy % on the training set.
output_directory = './results'

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

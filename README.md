# ssu_geosciences
Sonoma State University Geosciences - Koret Scholarship, Sigmaclast Classifier

![sigmaclast](http://searg.rhul.ac.uk/searg_uploads/2015/01/SER-21.jpg)

### Setup


##### Creating an environment from an environment.yml file 
##### Use the Terminal or an Anaconda Prompt for the following steps.


Create the environment from the environment.yml file:
```
conda env create -f environment.yml
```


Activate the new environment:
```
Windows: activate Keras
macOS and Linux: source activate Keras
```


Verify that the new environment was installed correctly:
```
conda list
```

### Config

  There are a few options that can be adjusted. To change settings open config.py in any text editor.
  ```
  emacs config.py
  ```
  or
  ```
  vim config.py
  ```
  
  Options that can be edited are as follows:
  ```
  model_name: the model names are listed in the config.py file, the only model that is custom is the SSUGeosciences model
  
  batch_size: this is the number of images processed per iteration in an epoch.
  
  num_epochs: number of times to train
  
  learning_rate: the smaller the number the longer it takes to learn. Too small of a number or too large of a number 
                 can cause our learning to be unsuccessful
                 
  ratio_train: since we do not break our images into train/dev/test sets we must do that dynamically. This is the % of
               our images that we want to use in our training set. ratio_dev is derived from ratio_train and ratio_test
  
  ratio_test: --CURRENTLY UNUSED AS OUR DATASET IS TOO SMALL-- this is the percent of our images that we will use to test upon. 
  
  image_directory: the directory that our images to binary classify our stored in. Our images must be stored in their own 
                   subdirectories. I.e. images/with and images/without. Only 2 directorie will be considered when the program 
                   is ran. 
  output_directory: This directory will store the output of our model. Every time the model is ran the results will be 
                     stored in output_directory/model_name/%.txt where % is the accuracy percent
                     
  optimizer: An optimizer is one of the two arguments required for compiling a Keras model. There are a few keras optimizer  
             options that are all listed in the config.py just above the optimizer definition.
  ```
    

### Use

  Activate the Keras environment
  ```
  source activate Keras
  ```
  
  Run the binary classifier
  ```
  python main.py
  ```
  
  Output is produced in multiple ways:
     1. The output is displayed to the user
     2. The config file used is copied and it's results are appended as a comment to the bottom of the file. 
        Additionally the file is copied to the results folder defined in the config.py file with it's name based on the
        accuracy of the script run.
        
        
 ### Visualize GPU Utilization
 ```
 watch -n 0.5 nvidia-smi
 ```
 
 ### Note on K-Fold cross Validation
 
 We use K-Fold cross val merely to test the validity of a model. After we choose a model, based on our cross validation
 we should then train our model on all of our data. 
 
 ```
 https://stats.stackexchange.com/questions/52274/how-to-choose-a-predictive-model-after-k-fold-cross-validation
 
 Say we have two models, say a linear regression model and a neural network. How can we say which model is 
 better? We can do K-fold cross-validation and see which one proves better at predicting the test set 
 points. But once we have used cross-validation to select the better performing model, we train that
 model (whether it be the linear regression or the neural network) on all the data. We don't use the 
 actual model instances we trained during cross-validation for our final predictive model.
 ```

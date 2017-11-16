
# coding: utf-8

# Zihao Mao
# 
# Cameron Matson
# 
# 9/22/2017
# 
# # Lab 3: Images
# 
# ## Introduction
# 
# For this lab we examine the images of the Stanford Dog Dataset.  The dataset consits of ~20,000 images of dogs from 120 different breeds.
# 
# #### Issues
# The dataset is primarily used for fine-grained classification problems, meaning that the instances are all members of the same main class and are divided by subclass.  In this case, the main class is 'Dog' and the subclass is the breed: 'Beagle', 'Poodle', 'Lab'...  These are potentially more difficult than standard classification problems because in theory all members of teh main class should at least share similar features.  In other words as the saying goes "a dog is a dog is a dog not a cat."
# 
# Another challenge with this dataset is that there is that they do not depict a standard scene.  These are not faces of dogs.  These are not photoshoot photos of dogs.  The images in the dataset are not even exclusively of dogs.  Some contain multiple dogs or even people.  The dataset would benefit from preprocessing in the form of some sort of standardization such that all the images are of the same kind, using facial detection for instance.
# 
# #### Uses
# We imagine one potential use for the finegrained classification of dogs could be used in searching for lost pets.  Imagine poor Susan has lost her precious Bichon Frise, Tutu.  She goes to her local police station and demands that they check all of the town's traffic cameras for traces of Tutu.  Well, they say there's hours of footage, and we don't want to look at it.  Poor Susan.  Now suppose there is a program that will "watch" the video and recognize when there is a four legged animal in view.  The image could then be put through a classifier to detect if that 4 legged beast is a dog or a cat (or something else).  Hooray!  It's a dog!  Now the image is put through a *fine-grained* classifier, which is able to tell that the dog **IS** in fact a Bichon Frise and not a Yorkshire Terrier.  The police are then able to determine where Tutu is and Susan is very happy.
# 
# #### Accuracy
# How well does a system like that need to work?  Well each successive level probably does not need to be as precise as the last (and it likely won't be cause each successive level is more difficult than the last.)  The key point is that a human (with some knowledge of dog breeds) would be close to perfect at identifying dogs, but with thousands of street cameras around, it would take them a long time to go through all the footage.  Assuming you do a good job of identifying the dogs in the image you probably don't have to be that accurate at identifing the bichon frise.  As long as you have as few false negatives as possible (so that you don't miss a potential bichon) you could probably get away with a few false positives.

# In[2]:


import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.misc import imresize
from skimage.color import rgb2gray
imagedir = '../data/dogs'


# ## Data Preprocessing
# 
# There are 120 different breeds included in the dataset with bout 150 images of each breed for a total of 20,580 images.  The images are stored in directories by breed.  To make the size of the dataset more managable, we'll take a sample of 50 images from each 60 of the breed.

# In[3]:


# remove dsstore
for d in os.listdir(imagedir):
    if d.find('.DS') != -1:
        os.remove(os.path.join(imagedir,d))
        continue
    for f in os.listdir(os.path.join(imagedir, d)):
        if f.find('.DS') != -1:
            os.remove(os.path.join(imagedir,d,f))
    


# In[33]:


def load_images(num_classes, h, w):
    
    # preinitialize the matrix
    #img_arr = np.empty((num_samples_per_breed*num_breeds,h*w))  # 20 instances of each breed, each img will be 200x200 = 40000 pixels
    img_arr = []
    label_arr = []
    
    # sample 60 breeds from the dataset
    a = np.arange(len(os.listdir(imagedir)))
    np.random.shuffle(a)
    breed_sample_idxs = a[:num_classes]
    for i, idx in enumerate(breed_sample_idxs):
        breed = os.listdir(imagedir)[idx]
        if breed[0] == '.' : 
            continue # stupid ds.store on mac
        print(i,breed)
        
        for img in os.listdir(os.path.join(imagedir, breed)):
            dog_path = os.path.join(imagedir,breed,img)            

            img = plt.imread(dog_path)
            
            # converts image to gray, resizes it to be 200x200, and then linearizes it
            img_gray_resize_flat = rgb2gray(imresize(img, (h,w,3))).flatten()
                        
            img_arr.append(img_gray_resize_flat)

            # add name to list of labels
            fname = dog_path.split('/')[-1] # 'dog_name_123497.jpg'
            dog_name = fname[:fname.rfind('_')] # 'dog_name'
            label_arr.append(breed)
            
    return img_arr, label_arr


# In[91]:

num_breeds = 5
h=128
w=128
dogs, labels = load_images(num_classes=num_breeds, h=h, w=w)
print(len(dogs))


# In[94]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

df= pd.DataFrame(dogs)

X = np.array(dogs)

enc = LabelEncoder()
y = enc.fit_transform(labels)

df


# In[95]:


ex = dogs[0].reshape((h,w))


# Here we go.

# In[96]:


from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(0) # using this to help make results reproducible

# Split it into train / test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Split X_train again to create validation data
#X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2)

X_train.shape


# In[44]:


import keras
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

keras.__version__


# In[97]:


NUM_CLASSES = 5
print(X_train.shape)
print(X_test.shape)


# In[98]:


y_train_ohe = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test_ohe = keras.utils.to_categorical(y_test, NUM_CLASSES)

# # make a 3 layer keras MLP\
# mlp = Sequential()
# mlp.add( Dense(input_dim=X_train.shape[1], units=30, activation='relu') )
# mlp.add( Dense(units=15, activation='relu') )
# mlp.add( Dense(NUM_CLASSES) )
# mlp.add( Activation('softmax') )

# mlp.compile(loss='mean_squared_error', 
#             optimizer='rmsprop',
#             metrics=['accuracy'])
                            
# mlp.fit(X_train, y_train_ohe,
#         batch_size=32, epochs=100,
#         shuffle=True, verbose=1)


# In[ ]:


cnn_layers = [16, 16]

# make a CNN with conv layer and max pooling
cnn = Sequential()
cnn.add(Reshape((1,h,w), input_shape=(1,h*w)))

for n in cnn_layers:
    cnn.add(Conv2D(filters=n, kernel_size= (3, 3), padding='same'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    
# add one layer on flattened output
cnn.add(Flatten())
cnn.add(Dense(50))
cnn.add(Activation('relu'))
cnn.add(Dense(NUM_CLASSES))
cnn.add(Activation('softmax'))

import time

t = time.time()

# Let's train the model 
cnn.compile(loss='mean_squared_error',
            optimizer='rmsprop',
            metrics=['accuracy'])

# we need to exapnd the dimensions here to give the
#   "channels" dimension expected by Keras
cnn.fit(np.expand_dims(X_train, axis=1), y_train_ohe,
        batch_size=32, epochs=100,
        shuffle=True, verbose=1)

print('time', time.time() - t)

# In[89]:


from sklearn import metrics as mt
from matplotlib import pyplot as plt
import seaborn as sns

def compare_mlp_cnn(cnn, mlp, X_test, y_test):
    if cnn is not None:
        yhat_cnn = np.argmax(cnn.predict(np.expand_dims(X_test, axis=1)), axis=1)
        acc_cnn = mt.accuracy_score(y_test,yhat_cnn)
        #plt.subplot(1,2,1)
        #cm = mt.confusion_matrix(y_test,yhat_cnn)
        #cm = cm/np.sum(cm,axis=1)[:,np.newaxis]
        #sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=enc.inverse_transform([0, 1, 2, 3, 4]), 
        #           yticklabels=enc.inverse_transform([0, 1, 2, 3, 4]))
        #plt.title('CNN: '+str(acc_cnn))
        print('cnn', acc_cnn)

    if mlp is not None:
        yhat_mlp = np.argmax(mlp.predict(X_test), axis=1)
        acc_mlp = mt.accuracy_score(y_test,yhat_mlp)
        #plt.subplot(1,2,2)
        #cm = mt.confusion_matrix(y_test,yhat_mlp)
        # cm = cm/np.sum(cm,axis=1)[:,np.newaxis]
        # sns.heatmap(cm,annot=True, fmt='.2f', xticklabels=enc.inverse_transform([0, 1, 2, 3, 4]), 
        #            yticklabels=enc.inverse_transform([0, 1, 2, 3, 4]))
        # plt.title('MLP: '+str(acc_mlp))
        print('mlp', acc_mlp)

# In[90]:


compare_mlp_cnn(cnn,mlp,X_test,y_test)


# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:27:25 2019

@author: YASH SAINI
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

clf=Sequential()
#clf.add(Convolution2D(input_shape=(64,64,3),filters=32,kernel_size=[3,3],strides=1,activation='relu'))

''' 1) Convolution'''
clf.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))

''' 2) Pooling'''
clf.add(MaxPooling2D(pool_size=(2,2)))


''' Addition of new convolutional layer enables higher accuracy'''
clf.add(Convolution2D(32,(3,3),activation='relu'))
clf.add(MaxPooling2D(pool_size=(2,2)))

''' 3) Flattening ( 1d array of features)'''
clf.add(Flatten())

'''4) Full Connection (addition of hidden and output layers)
Old method:-
clf.add(Dense(output_dim=128,activation='relu'))
clf.add(Dense(output_dim=1,activation='sigmoid'))'''

clf.add(Dense(units=128,kernel_initializer='uniform',activation='relu'))
clf.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

#compiling ANN
clf.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])#adam for stochastic gradient descent


#Fitting image
''' Open keras documentation on browser select image preprocessing .Augmentation helps to enrich 
preprocessing and reduces overfitting. Patches of images are formed'''

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set1 = train_datagen.flow_from_directory(
        'training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set1 = test_datagen.flow_from_directory(
        'test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

clf.fit_generator(
        training_set1,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set1,
        validation_steps=2000) 




''' Making New predictions'''
#making single predicitons
import numpy as np
from keras.preprocessing import image
test_image=image.load_img("single_prediction/cat_or_dog_1.jpg",target_size=(64, 64))
test_image=image.img_to_array(test_image)

# add a new dimension

test_image=np.expand_dims(test_image,axis=0)
result=clf.predict(test_image)
print training_set1.class_indices
if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'

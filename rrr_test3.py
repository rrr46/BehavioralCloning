import numpy as np 
import cv2
import csv
import sklearn
from sklearn.utils import shuffle
import scipy
import matplotlib as plt

samples = []
with open('August11th/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print(np.shape(train_samples))

def intensity_change(image):
	im = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
	im = np.array(im, dtype = np.float64)
	rv = 0.5 + np.random.uniform()
	im[:,:,2] = im[:,:,2]*rv
	im[:,:,2][im[:,:,2] > 255] = 255
	im = np.array(im, dtype = np.uint8)
	im = cv2.cvtColor(im,cv2.COLOR_HSV2RGB)
	return im

def view_change(image,s,trange):
	xc = trange*np.random.uniform() - trange/2
	sang = s + xc/trange*2*.2
	yx = 40*np.random.uniform() - 40/2
	TM =np.float32([[1,0,xc],[0,1,yx]])
	image_vc = cv2.warpAffine(image,TM,(160,320))
	# print(np.shape(image_vc))
	return image_vc,sang

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
            	#
                #name = 'C:/Users/rranade/Documents/GitHub/CarND-Behavioral-Cloning-P3/windows_sim/windows_sim/windows_sim_Data/IMG/August9th/IMG'+batch_sample[0].split('/')[-1]
                name = batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                print(np.shape(center_image))
                #center_image = center_image[50:140,:,:]
                center_angle = float(batch_sample[3])
                #center_image = scipy.misc.imresize(center_image,(64,64,3))
                images.append(center_image)
               	angles.append(center_angle)

               	# image_vc,angle_vc = view_change(center_image,center_angle,100)
               	# print(np.shape(image_vc))
               	# images.append(image_vc)
               	# angles.append(angle_vc)
                # image_flip = np.fliplr(center_image)
                # images.append(image_flip)
                # angles.append(center_angle*-1.0)

                # image_ic = intensity_change(center_image)
                # #image_ic = scipy.misc.imresize(image_ic,(64,64,3))
                # images.append(image_ic)
                # angles.append(center_angle)

                name = batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                #left_image = scipy.misc.imresize(left_image,(64,64,3))
                #left_image = left_image[50:140,:,:]
                #left_image = cv2.resize(left_image,(64,64,3))
                images.append(left_image)
                angles.append(center_angle + 0.25)

                # image_il = intensity_change(left_image)
                # #image_il = scipy.misc.imresize(image_il,(64,64,3))
                # images.append(image_il)
                # angles.append(center_angle + 0.25)

                name = batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                #right_image = scipy.misc.imresize(right_image,(64,64,3))
                #right_image = right_image[50:140,:,:]
                #right_image = cv2.resize(right_image,(64,64,3))
                images.append(right_image)
                angles.append(center_angle - 0.25)

                # image_ir = intensity_change(right_image)
                # #image_ir = scipy.misc.imresize(image_ir,(64,64,3))
                # images.append(image_ir)
                # angles.append(center_angle - 0.25)
                #print(batch_sample[1])
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            print(np.size(X_train))
            print(np.size(y_train))
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

def resize_image(image):
	import tensorflow as tf
	return tf.image.resize_images(image,(64,64))
#print('Test')
#print (np.size(train_samples))
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Reshape
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda x : (x/255.0 - 0.5), input_shape = (160,320,3)))
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(resize_image, name = 'ResizeImage'))
#model.add(Lambda(lambda x : (x/255.0), input_shape = (64,64,3)))
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
# model.add(Lambda(lambda x : tf.image.resize_images(x,(64,64))))
#model.add(Reshape((64,64,3), input_shape = (90,320,3)))
#model.add(Lambda(lambda x : (x/255.0), input_shape = (90,320,3)))
##model.add(Convolution2D(3,3,3,activation = 'relu')) #TODO
#Input = 160x320x3. Output = 166x316x6
##model.add(Convolution2D(32,5,5,activation = 'relu')) #TODO
#Input = 86x316x16. Output = 83x158x6
#model.add(MaxPooling2D(pool_size = (2,2))) #TODO
##model.add(Convolution2D(32,2,2,activation = 'relu')) #TODO
#Input = 83x158x6. Output = 79x154x16 
##model.add(Convolution2D(64,5,5,activation = 'relu'))
#Input= 79x154x16. Output = 38x77x16
##model.add(MaxPooling2D(pool_size = (2,2)))
#Input = 38x77x16.
##model.add(Convolution2D(128,5,5,activation = 'relu'))
##model.add(Convolution2D(256,5,5,activation = 'relu'))
#model.add(Dropout(0.5))
##model.add(Flatten())
##model.add(Dense(1024, activation = 'relu'))
#model.add(Dropout(0.5))
##model.add(Dense(512, activation = 'relu'))
##model.add(Dropout(0.5))
##model.add(Dense(64, activation = 'relu'))
##model.add(Dense(1))
##Model 2
# model.add(Conv2D(3,kernel_size = (5,5),activation = 'relu',use_bias=True, input_shape = (32,32,3)))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Conv2D(24,kernel_size = (5,5),activation = 'relu',use_bias=True))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Conv2D(96,kernel_size = (5,5),activation = 'relu',use_bias=True))
# model.add(Flatten())
# model.add(Dense(1024,activation = 'relu',use_bias = True))
# model.add(Dropout(0.5))
# model.add(Dense(256,activation = 'relu',use_bias = True))
# model.add(Dense(32,activation = 'relu',use_bias = True))
# model.add(Dropout(0.5))
# model.add(Dense(1))

#Model 3
model.add(Conv2D(3,kernel_size = (1,1),activation = 'relu',use_bias=False, input_shape = (64,64,3)))
model.add(Conv2D(32,kernel_size = (3,3),activation = 'relu',use_bias=False))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64,kernel_size = (3,3),activation = 'relu',use_bias=False))
model.add(Conv2D(64,kernel_size = (3,3),activation = 'relu',use_bias=False))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(128,kernel_size = (3,3),activation = 'relu',use_bias=False))
model.add(Conv2D(128,kernel_size = (3,3),activation = 'relu',use_bias=False))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512,activation = 'relu',use_bias = False))
model.add(Dense(64,activation = 'relu',use_bias = False))
model.add(Dense(16,activation = 'relu',use_bias = False))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)/1, validation_data=validation_generator, nb_val_samples=len(validation_samples)/1, nb_epoch=2)
model.save('model_test3.h5')
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train','test'])

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train','test'])

plt.show()
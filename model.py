'''
=======================================================================================
Author: Chun-Wei Chiang
Date: 2019.04.08
Description: Train Siamese network
=======================================================================================
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, concatenate, MaxPool2D, Lambda, Flatten, Dense,  GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import get_file
import tensorflow.keras.backend as K
import pickle

sqz = 'sqz1'
exp = 'exp'
exp1 = 'exp1'
exp3 = 'exp3'
relu = 'relu_'
models_dir = "models"
history_dir = "history"
checkpoint_dir = "checkpoint"
model_name ="siameseNet"


def fire_module(prv_lyr, fire_id, squeeze = 3, expand = 4):
	s_id = 'fire' + str(fire_id) + '/'

	#squeeze layer
	sqz_layer = Conv2D( squeeze, kernel_size=(1,1), padding='same', name=s_id+sqz )(prv_lyr)
	sqz_layer = Activation( 'relu', name=s_id+relu+sqz )(sqz_layer)

	#expand layer
	#1*1
	exp1_layer = Conv2D( expand, kernel_size=(1,1), padding='same', name=s_id+exp1)(sqz_layer)
	exp1_layer = Activation( 'relu', name=s_id+relu+exp1)(exp1_layer)
	#3*3
	exp3_layer = Conv2D( expand, kernel_size=(3,3), padding='same', name=s_id+exp3)(sqz_layer)
	exp3_layer = Activation( 'relu', name=s_id+relu+exp3)(exp3_layer)

	cnct_layer = concatenate([exp1_layer, exp3_layer])

	return cnct_layer

def squeezeNet():
	
	inputs = Input(shape=(32,32,3))

	x = Conv2D(96, kernel_size=(4,4), padding='same', name='conv1' )(inputs)
	x = Activation('relu', name='relu_conv1')(x)
	x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)

	x = fire_module(x, fire_id=2, squeeze=16, expand=64)
	x = fire_module(x, fire_id=3, squeeze=16, expand=64)
	x = fire_module(x, fire_id=4, squeeze=32, expand=128)		
	x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='pool2')(x)

	x = fire_module(x, fire_id=5, squeeze=32, expand=128)
	x = fire_module(x, fire_id=6, squeeze=48, expand=192)
	x = fire_module(x, fire_id=7, squeeze=48, expand=192)
	x = fire_module(x, fire_id=8, squeeze=64, expand=256)
	x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='pool3')(x)

	x = fire_module(x, fire_id=9, squeeze=64, expand=256)
	x = BatchNormalization()(x)
	x = Conv2D(10, kernel_size=(4,4), padding='same', name='conv10')(x)
	x = Activation('relu', name='relu_conv10')(x)

	x = GlobalAveragePooling2D()(x)
	model = Model(
		inputs = inputs,
		outputs = x
		)
	# x = Flatten()(x)
	# x = Dense(10)(x)
	# Remove the softmax layers
	# x = Activation('softmax', name='softmax')(x)
			
	return model

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shapes[0], shapes[2])

def siameseNet():
	input_base = Input(shape=(32,32,3))
	input_pair = Input(shape=(32,32,3))

	basemodel = squeezeNet()
	encode_base = basemodel(input_base)
	encode_pair = basemodel(input_pair)

	L2_layer = Lambda( lambda tensor: K.sqrt(K.sum((tensor[0]-tensor[1])**2, axis=1, keepdims=True )),  output_shape=eucl_dist_output_shape)
	L2_distance = L2_layer([encode_base, encode_pair])
	# pred_label = Activation( 'sigmoid')(L2_distance)

	model = Model(
			inputs = [input_base, input_pair],
	 		outputs= L2_distance,
	 		)

	return model


def siameseLoss(yTrue, yPred):
	# yTrue is label, and yPred is distance
	return K.mean( (1-yTrue) * K.square(yPred) + yTrue * K.square(K.maximum(1 - yPred, 0)))


def accuracy(yTrue, yPred):
	# print(yPred.shape)
	# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	return K.mean(K.equal(yTrue,K.cast(yPred > 0.5, yTrue.dtype)))
	# return K.mean(yTrue)

def trainModel(train_data, src_img, epochs=200, learning_rate=0.001 ):
	model = None
	h5_storage_path = models_dir + "/" + model_name + str(learning_rate) + ".h5"
	hist_storage_path = history_dir + "/" + model_name + str(learning_rate)
	checkpoint_path = checkpoint_dir + "/" + model_name + str(learning_rate) + ".hdf5"

	model = siameseNet()
	model.compile(
		loss = siameseLoss,
		optimizer = SGD(lr = learning_rate),
		metrics= [accuracy]
		)

	checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]

	train_base = [src_img[i[0]] for i in train_data]
	train_pair = [src_img[i[1]] for i in train_data]
	train_label = np.array([ i[2] for i in train_data ])

	# print(src_img.shape)
	# print(len(train_data[:]))
	print("!!!!!")
	print(train_label.shape)

	#Fit the model
	hist = model.fit(
		[train_base, train_pair],
		train_label,
		epochs = epochs,
		batch_size = 32,
		# validation_data=(validation_data, validation_label),
		validation_split = 0.3,
		callbacks=callbacks_list,
		verbose= 1)

	#save the model
	save_model(
		model,
		h5_storage_path,
		overwrite=True,
		include_optimizer=True
	)

	#Save the history of training
	with open(hist_storage_path, 'wb') as file_hist:
		pickle.dump(hist.history, file_hist)

	print("Successfully save the model at " + h5_storage_path)

	return model

def loadModel(learning_rate = 0.001):
	h5_storage_path = models_dir + "/" + model_name + str(learning_rate) + ".h5"
	
	try:
		model = load_model(
			h5_storage_path,
			custom_objects={'siameseLoss':siameseLoss},
			compile=True
		)

	except Exception as e:
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		model = None
		print(e)
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")

	finally:
		return model
'''
=======================================================================================
Author: Chun-Wei Chiang
Date: 2019.04.08
Description: Train Siamese network
=======================================================================================
'''
import tensorflow as tf
import model
import util
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow.keras.utils import to_categorical
from collections import defaultdict
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow.keras.backend as K

cifar_10_dir = "cifar-10"
validation_number = 10000
load_max_number = 50000
train_number = load_max_number - validation_number

epochs = 50
learning_rate = 0.01

def testModel(model, x_test, y_test, learning_rate):
	score = model.evaluate(x_test, y_test)
	print("=============================================")
	print("Test perforemance of learning rate " + str(learning_rate))
	print('Test loss:'+ str(score[0]))
	print('Test accuracy:'+ str(score[1]))


def plot_performance(histories, name_list, isloss = True, isVal = False, isBoth = False):
	#isloss means whether plot loss. If True, plot loss, nor plot accuracy

	perforemance = 'loss' if isloss else 'accuracy'

	# print(perforemance)
	fig = plt.figure()

	for hist in histories:
		if isBoth:
			plt.plot(hist[perforemance])
			plt.plot(hist['val_' + perforemance])
		else:
			val = 'val_' if isVal else ''
			perforemance = val + perforemance
			plt.plot(hist[perforemance])

	plt.xticks(np.arange(0, epochs +1 , epochs/5 ))
	plt.ylabel(perforemance)
	plt.xlabel( "epochs" )
	plt.legend( name_list , loc=0)
	# plt.show()
	fig.savefig(perforemance + '.png')

def compute_accuracy(y_true, y_pred):
	correct = 0
	for x in range(len(y_true)):
		pred_label = 0 if y_pred[x] < 0.5 else 1
		if pred_label == y_true[x]:
			correct += 1

	return correct/len(y_true)


if __name__ == '__main__':
	train_data, _, train_labels, test_data, _, test_labels, label_names = util.loadCIFAR10(cifar_10_dir)

	label_dict = {}
	label_dict = defaultdict(lambda: [], label_dict)

	train_labels = train_labels[ 0 : load_max_number ]

	for idx, value in np.ndenumerate(train_labels):
		label_dict[value].append(idx)

	# print(list(label_dict)[0])

	train_data, test_data = train_data/255.0, test_data/255.0
	train_idx_label = []

	for idx, y in enumerate(list(label_dict)):
		for x in range(3000):
			cifar_label = y
			# print(len(label_dict[6]))
			train_base = label_dict[cifar_label][random.randint(0, 4999)]
			train_pair = label_dict[cifar_label][random.randint(0, 4999)]
			true_pair = (train_base, train_pair, 0, idx, idx)
			train_idx_label.append(true_pair)
			
		for x in range(3000):
			cifar_label = y
			index_range = list(range(0, idx)) + list(range(idx + 1, 10))
			ramdom_idx = random.choice(index_range)
			cifar_imposter_label = list(label_dict)[ramdom_idx]
			train_base = label_dict[cifar_label][random.randint(0, 4999)]
			train_pair = label_dict[cifar_imposter_label][random.randint(0, 4999)]
			true_pair = (train_base, train_pair, 1, idx, ramdom_idx)
			train_idx_label.append(true_pair)

	random.shuffle(train_idx_label)
	print(len(train_idx_label))

	

	config = ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		tf.set_random_seed(1)
		Siamese = model.loadModel(learning_rate = learning_rate)
		if Siamese == None:
			#label_dict.keys()[0]
			Siamese = model.trainModel(train_idx_label, train_data, epochs = epochs, learning_rate = learning_rate)
		
		#Test the model
		test_base = [train_data[i[0]] for i in train_idx_label]
		test_pair = [train_data[i[1]] for i in train_idx_label]
		test_label = np.array([ i[2] for i in train_idx_label ])

		print(test_label)
		predictions = Siamese.predict([test_base, test_pair]).flatten()
		print(predictions)
		test_acc = compute_accuracy(test_label, predictions)
		print("accuracy")
		print(test_acc)

		total = np.zeros((10, 10))
		correct = np.zeros((10, 10))
		acc_metrix = np.zeros((10, 10))

		for idx, x in enumerate(train_idx_label):
			a_label = x[3]
			b_label = x[4]
			pred = predictions[idx]
			truth = test_label[idx]

			total[a_label][b_label] += 1
			total[b_label][a_label] += 1

			pred_label = 0 if pred < 0.5 else 1
		
			if pred_label == truth:
				correct[a_label][b_label] += 1 
				correct[b_label][a_label] += 1

		for x in range(10):
			for y in range(10):
				acc_metrix[x][y] = correct[x][y]/total[x][y]

		print(acc_metrix)
		

		# histories = []
		# name_list = []
		# path = "history/siameseNet0.01"
		# histories.append( util.unpickle(path) )
		# name = "siameseNet"
		# name_list.append( name )
		# name_list.append( name + "_val" )

		# plot_performance(histories, name_list, isloss = False, isVal=False, isBoth = True)
		# pred_label = K.less(0.5, )
		# print(pred_label.shape)
		# confusion = tf.contrib.metrics.confusion_matrix(labels=truth_label, predictions=pred_label, num_classes=2)


# ===============================================================================================================
	# train_labels= to_categorical(train_labels,num_classes=10)
	# validation_labels = to_categorical(validation_labels, num_classes=10)
	# test_labels= to_categorical(test_labels,num_classes=10)

	# learning_rate_list = [0.01, 0.001, 0.0001]
	# histories = []
	# name_list = []

	# for x in learning_rate_list:
	# 	with tf.Session() as sess:
	# 		tf.set_random_seed(1)
	# 		squeezenet = model.loadModel(learning_rate = x)
	# 		if squeezenet == None: 
	# 			squeezenet = model.trainModel(train_data, train_labels, validation_data, validation_labels, epochs=epochs, learning_rate=x)
			
	# 		pred_label = np.argmax(squeezenet.predict(test_data), axis=1)
	# 		truth_label = np.argmax(test_labels, axis=1)
	# 		# testModel(squeezenet, test_data, test_labels, x)
	# 		# path = "history/squeezeNet_{}".format(x)
	# 		# histories.append( util.unpickle(path) )
	# 		# name = "{}".format(x)
	# 		# name_list.append( name )
	# 		# name_list.append( name + "_val" )

	# 		print(x)
	# 		confusion = tf.contrib.metrics.confusion_matrix(labels=truth_label, predictions=pred_label, num_classes=10)
	# 		print(confusion.eval(session=sess))


	# plot_performance(histories, name_list, isloss = True, isBoth = True)



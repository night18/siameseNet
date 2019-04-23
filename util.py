'''
=======================================================================================
Author: Chun-Wei Chiang
Date: 2019.04.08
Description: Load CIFAR-10 Dataset and print the images.
=======================================================================================
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle


def unpickle(file):
	'''
	load the CIFAR-10 data
	here is the source of CIFAR-10 datasets: https://www.cs.toronto.edu/~kriz/cifar.html
	'''
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict




def loadCIFAR10(data_dir):
	'''
	return the following information
	cifar_train_data: 		ndarray (50000, 32, 32, 3)
	cifar_train_filenames: 	ndarray (50000,)
	cifar_train_labels: 	ndarray (50000,)
	cifar_test_data: 		ndarray (10000, 32, 32, 3)
	cifar_test_filenames:	ndarray (10000,)
	cifar_test_labels: 		ndarray (10000,)
	cifar_label_names:		array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], dtype='|S10')
	'''

	'''
	get the meta data dictionary
	{
	'label_names': ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'],
	'num_cases_per_batch': 10000,
	'num_vis': 3072
	}
	'''
	meta_data_dict = unpickle(data_dir + "/batches.meta")
	cifar_label_names = np.array(meta_data_dict[b'label_names'])

	'''
	get training data
	(50000, 3072)
	'''
	cifar_train_data = None
	cifar_train_filenames = []
	cifar_train_labels = []

	for x in range(1,6):
		cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(x))
		if x == 1:
			cifar_train_data = cifar_train_data_dict[b'data']
		else:	
			cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
		
		# it will become 2D list if using append() here.
		cifar_train_filenames += cifar_train_data_dict[b'filenames']
		cifar_train_labels += cifar_train_data_dict[b'labels']
		
	'''
	reshape the array to image like (3,32,32)
	The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue. 
	The values are stored in row-major order, so the first 32 bytes are the red channel values of the first row of the image. 
	(50000, 3072) => (10000, 3, 32, 32)
	'''
	cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))

	'''
	change the axis sequence
	move the color axis to the last one
	'''
	cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
	cifar_train_filenames = np.array(cifar_train_filenames)
	cifar_train_labels = np.array(cifar_train_labels)

	'''
	get testing data
	(10000, 3072) => (10000, 3, 32, 32)
	'''
	cifar_test_data_dict = unpickle(data_dir + "/test_batch")
	cifar_test_data = cifar_test_data_dict[b'data']
	cifar_test_filenames = cifar_test_data_dict[b'filenames']
	cifar_test_labels = cifar_test_data_dict[b'labels']
	
	cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
	'''
	change the axis sequence
	move the color axis to the last one
	'''
	cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
	cifar_test_filenames = np.array(cifar_test_filenames)
	cifar_test_labels = np.array(cifar_test_labels)

	return cifar_train_data, cifar_train_filenames, cifar_train_labels, cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names


if __name__ == "__main__":
	'''
	testing
	'''

	cifar_10_dir = "cifar-10"

	train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = loadCIFAR10(cifar_10_dir)

	num_plot = 5
	f, ax = plt.subplots(num_plot, num_plot)
	for m in range(num_plot):
		for n in range(num_plot):
			idx = np.random.randint(0, train_data.shape[0])
			ax[m, n].imshow(train_data[idx])
			ax[m, n].get_xaxis().set_visible(False)
			ax[m, n].get_yaxis().set_visible(False)
	f.subplots_adjust(hspace=0.1)
	f.subplots_adjust(wspace=0)
	plt.show()
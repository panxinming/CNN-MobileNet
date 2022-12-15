import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

#data loader for tiny-imagenet-200, the source is https://image-net.org/download-images.php
#tiny-imagenet-200 data set has 500 images for each class in the training data, and the training data have 200 classes.
#For validation data, it has 50 images for each class and the total number of classes is 200.


tiny_img200_label = {}

def load_data(dataset, stage, dir):
	image_list = []
	label_list = []
	if(dataset=='tiny-imagenet-200'):
		'''
		For tiny-imagenet-200
		Each sub_dir is a folder, and the folder's name is the class (Y). 
		All the files in the folder is images (our training data X).
		'''

		if(stage=='train'):
			label_id = 0
			#loading training images and labels
			for sub_dir in os.listdir(dir):
				tiny_img200_label[sub_dir] = label_id
				label_id += 1
				sub_dir_name=os.path.join(dir, sub_dir)
				sub_dir_name=os.path.join(sub_dir_name, "images")
			    #print("Reading folder {}".format(sub_dir))
				for file in os.listdir(sub_dir_name):
					filename = os.fsdecode(file)
					image_list.append(np.array(Image.open(os.path.join(sub_dir_name,file))))
			        #here we only save the matrix in the list rather than the original image file
					label_list.append(str(sub_dir))
			'''
			The tiny-imagenet-200 data set don't give labels for test data, so we cannot
			use them for experiments. Instead, we use the validation data set as our test data here, 
			and we split some data from the training data set to be the validation data set. 

			Besides, the images in tiny-imagenet-200 are 64*64. There are two problems.
			First, mobilenet take the input shape (224,224,3), however, 
			the data have a shape of (64,64,3). Therefore, we need to resize them.
			Second, some data have the shape (64,64,1). We calculate the proportion of these data,
			and we find that they only accounts for fewer than 2% (both in the training data set and the validation data set)
			Therefore, we decide to eliminate these data.
			'''
			train_data_list, train_label_list, val_data_list, val_label_list = split_trainset(dataset,image_list,label_list)
			X_train, y_train = clean_data(dataset, train_data_list, train_label_list)
			X_val, y_val = clean_data(dataset, val_data_list, val_label_list)
			return X_train, y_train, X_val, y_val

		elif(stage=='test'):
			'''
			Here we use the data in the validation data folder as our test data set,
			because we don't have labels for the test data from the source we download it. 
			'''
			#loading validation images
			val_image_dir = os.path.join(dir,"images")
			for file in os.listdir(val_image_dir):
				filename = os.fsdecode(file)
				image_list.append(np.array(Image.open(os.path.join(val_image_dir,file))))

			#loading corresponding labels
			val_label_file = os.path.join(dir,"val_annotations.txt")
			val_df = pd.read_table(val_label_file, header = None, 
				names = ['filename','label','X','Y','H','W']) 
			#here we use read_table to save the txt file's content as a dataframe
			label_list = val_df['label']
			#for the same reason above, we need to call the clean_data function
			X_test, y_test = clean_data(dataset, image_list,label_list)
			return X_test, y_test

def split_trainset(dataset,train_data,train_labels):
	'''
	we call this function to split the training data into train data set and validation data set
	'''
	train_data_list = []
	train_label_list = []
	val_data_list = []
	val_label_list = []
	if(dataset == 'tiny-imagenet-200'):
		#every class have 500 images, we split 50 for validation
		cnt = 0
		for i in range(len(train_data)):
			cnt = cnt+1
			if(cnt<=450):
				train_data_list.append(train_data[i])
				train_label_list.append(train_labels[i])

			elif(cnt<=500):
				val_data_list.append(train_data[i])
				val_label_list.append(train_labels[i])
			
			else: # when cnt reaches 501, it means a new class, 
				cnt = 0
		return train_data_list, train_label_list, val_data_list, val_label_list





def cal_prop(dataset, image_list):
	'''
	this function is used to count the number of gray photos that don't satisfy the 
	input shape of mobilenet. It also calculates the proportion of these data.
	'''
	if(dataset=='tiny-imagenet-200'):
		single_channel_ind = []
		standard_shape = (64,64,3)
		for i in range(len(image_list)):
			if(image_list[i].shape!=standard_shape):
				single_channel_ind.append(i)
		print("The total number of single channel images in the training images is ",
			len(single_channel_ind), " ,and the proportion of them is ", 
			len(single_channel_ind)/len(image_list))
		return single_channel_ind 


def clean_data(dataset, image_list, label_list):
	'''
	This function first delete the single channel data and their labels,
	and then it resize the images to (224,224,3) if necessary
	'''

	single_channel_ind = cal_prop(dataset, image_list)
	add_ = int(224-image_list[0].shape[0]/2) 
	three_channel_images = []
	clean_label_list = []
	for i in range(len(image_list)):
		if(i in single_channel_ind):
			continue
		else:
	        #we need resize our images' shape into (224,224,3) to fit mobilenet's input size
	        
	        #Method 1: Image.resize()
			img = Image.fromarray(image_list[i])
			img = img.resize((224,224),Image.BILINEAR)
	        
	#         #Method 2: zero padding
	#         img = image_list[0]
	#         img_pad = np.zeros((224,224,3),dtype = int)
	#         for i in range(3):
	#             img_pad = np.pad(img[:,:,i],((add_,add_),(add_,add_)),
	#                             'constant',constant_values = (0))
	        
			three_channel_images.append(np.array(img))
			clean_label_list.append(tiny_img200_label[label_list[i]])
	X = np.array(three_channel_images)
	Y = np.array(clean_label_list)
	#Y = tf.one_hot(Y, depth = 200)
	print(X.shape, Y.shape)
	return X, Y

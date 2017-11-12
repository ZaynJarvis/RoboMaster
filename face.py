import cv2
import numpy as np 
import model as M 
import tensorflow as tf
import random 
def read_data():
	data = []
	f = open('trainlist.txt')
	# lin = f.readline()
	# info = lin.strip().split('\t')
	# x = info[1:]
	# print(info)
	for line in f:
		info = line.strip().split('\t')
		img_path = info[0]
		img = cv2.imread(img_path,0)
		img = cv2.resize(img,(128,128))
		img = img.reshape([128,128,1])
		# _, x1 , x2 , y1 , y2 = info
		data_row = [img]
		data_row.append(info[1:])
		# print(data_row)
		# input('..')
		data.append(data_row)
	# print(data[0])
	# print(data[1])
	# print(data[2])
	return data
		# print(img.shape)
		# cv2.imshow('Image',img)
		# cv2.waitKey(0)
		# print(data)
def build_model(input_placeholder):
	mod = M.Model(input_placeholder,[None,128,128,1])
	# mod = M.Model(input_placeholder, [None, 784]) #PEOPLE, -1 means None
	# mod.reshape([-1,128,128,1])# last param is channel
	mod.convLayer(5, 16, stride = 2, activation=M.PARAM_RELU) # 5 is kernel size, 6 is channel
	# mod.maxpoolLayer(2)                   #samples a 2x2 area 
	mod.convLayer(4, 32, stride = 2,activation=M.PARAM_RELU)
	# mod.maxpoolLayer(2)
	mod.convLayer(3, 64, stride = 2,activation=M.PARAM_RELU)
	mod.convLayer(3, 128, stride = 2,activation=M.PARAM_RELU)
	# mod.maxpoolLayer(2)
	mod.convLayer(3, 256, stride = 2,activation=M.PARAM_RELU)
	# mod.maxpoolLayer(2)
	mod.convLayer(3, 256*2,stride = 2, activation=M.PARAM_RELU)
	mod.flatten()
	# mod.fcLayer(50, activation=M.PARAM_RELU) 
	mod.fcLayer(100, activation=M.PARAM_RELU) 
	mod.fcLayer(50, activation=M.PARAM_RELU) 
	mod.fcLayer(4) #no need for activation;output layer always has no activation function
	return mod.get_current_layer()


def build_graph():
	input_placeholder = tf.placeholder(tf.float32,[None,128,128,1]) # 784 is 28 x 28 pixels, None means a arbitrary number of pictures. 784 is the number of pixels in each picture
	label_placeholder = tf.placeholder(tf.float32,[None,4]) # None means an number of pictures. 10 digits: '0' to '9'. There are 10 possible answers for each picture
	output = build_model(input_placeholder)
	loss = tf.reduce_mean(tf.square(output - label_placeholder))
	# accuracy = M.accuracy(output, tf.argmax(label_placeholder,1))
	train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
	return input_placeholder, label_placeholder, output, loss, train_step 

MAX_ITER = 10000
BSIZE = 16
input_placeholder, label_placeholder, output, loss, train_step = build_graph()
data = read_data()
with tf.Session() as sess:
	# saver = tf.train.Saver()
	M.loadSess('./model/', sess, init = True) # ./model/ is directory of savings
	for i in range(MAX_ITER):
		databatch = random.sample(data,BSIZE)
		img_batch , xy_batch = [i[0] for i in databatch],[i[1] for i in databatch]
		feed_d = {input_placeholder:img_batch, label_placeholder:xy_batch}


		# x_train, y_train = mnist.train.next_batch(128)
		ls, _ = sess.run([loss,train_step], feed_dict=feed_d)
		if i % 100 == 0:
			print('Iter : ',i,'\t| loss : ',ls)
		# if i % 2000 == 0:
			# saver.save(sess, './model/digit.ckpt')
			# acc = sess.run(accuracy,feed_dict={input_placeholder:mnist.test.images, label_placeholder:mnist.test.labels})
			# print('Test Accuracy : ', acc)
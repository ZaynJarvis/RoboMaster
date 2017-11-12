import tensorflow as tf
import numpy as np
import model as M
from tensorflow.examples.tutorials.mnist import input_data 
import cv2 as cv #                                         Hansel: "WHOEVER IS CODING, PLS USE TABS ONLY!!!!!!!!!!!"

# create data reader #replace function read_data
mnist = input_data.read_data_sets('MNIST_data',  one_hot = True)

def build_model(input_data):
	mod = M.Model(input_data, [None, 784]) #PEOPLE, -1 means None
	mod.reshape([-1,28,28,1])#No need to input -1 # last param is channel
	mod.convLayer(5, 6, activation=M.PARAM_RELU) # 5 is kernel size, 6 is channel
	mod.maxpoolLayer(2)                   #samples a 2x2 area 
	mod.convLayer(5, 16, activation=M.PARAM_RELU)
	mod.maxpoolLayer(2)
	mod.flatten()
	# mod.fcLayer(50, activation=M.PARAM_RELU) 
	mod.fcLayer(50, activation=M.PARAM_RELU) 
	mod.fcLayer(10) #no need for activation;output layer always has no activation function
	return mod.get_current_layer()

def build_graph():
	input_placeholder = tf.placeholder(tf.float32,[None,784]) # 784 is 28 x 28 pixels, None means a arbitrary number of pictures. 784 is the number of pixels in each picture
	label_placeholder = tf.placeholder(tf.float32,[None,10]) # None means an number of pictures. 10 digits: '0' to '9'. There are 10 possible answers for each picture
	output = build_model(input_placeholder)
	loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_placeholder, logits=output))
	accuracy = M.accuracy(output, tf.argmax(label_placeholder,1))
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	return input_placeholder, label_placeholder, output, loss, accuracy, train_step 

MAX_ITER = 100000
input_placeholder, label_placeholder, output, loss, accuracy, train_step = build_graph()

with tf.Session() as sess:
	saver = tf.train.Saver()
	M.loadSess('./model/', sess, init = True) # ./model/ is directory of savings
	for i in range(MAX_ITER):	
		x_train, y_train = mnist.train.next_batch(128)
		ls, acc, _ = sess.run([loss,accuracy,train_step], feed_dict={input_placeholder:x_train, label_placeholder:y_train})
		if i % 100 == 0:
			print('Iter : ',i,'\t| acc : ',acc,'\t| loss : ',ls)
		if i % 1000 == 0:
			acc = sess.run(accuracy,feed_dict={input_placeholder:mnist.test.images, label_placeholder:mnist.test.labels})
			print('Test Accuracy : ', acc)

	saver.save(sess, './model/digit.ckpt')

# No bug version
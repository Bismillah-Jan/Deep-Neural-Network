#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 23:26:25 2017
@author: bismillah
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def Network_model(input_data):
    
    """
    tensorflow models take inputs in the form of dictionary, so lets initialize the weights and biases 
    as given below.
    """
    layer1_param={'weights':tf.Variable(tf.random_normal([784, no_neurons_layer1])), 
    'biases': tf.Variable(tf.random_normal([no_neurons_layer1]))}
    
    layer2_param={'weights':tf.Variable(tf.random_normal([no_neurons_layer1, no_neurons_layer2])), 
    'biases': tf.Variable(tf.random_normal([no_neurons_layer2]))}
    
    layer3_param={'weights':tf.Variable(tf.random_normal([no_neurons_layer2, no_neurons_layer3])), 
    'biases': tf.Variable(tf.random_normal([no_neurons_layer3]))}
    
    layer4_param={'weights':tf.Variable(tf.random_normal([no_neurons_layer3, no_neurons_layer4])), 
    'biases': tf.Variable(tf.random_normal([no_neurons_layer4]))}
    
    output_layer_param={'weights':tf.Variable(tf.random_normal([no_neurons_layer4, no_classes])), 
    'biases': tf.Variable(tf.random_normal([no_classes]))}
    
    #so uptill now the weights for each layer is initialized
    
    """
    Now what will happened in each layer, I will define next. basically the weights are multiplied
    in each layer with the corresponding inputs and then it is passed through activation function 
    (relu in this case) and the output is given as input to the other layer.
    sign:B-Jan
    """
    
    l1_output= tf.add(tf.matmul(input_data,layer1_param['weights']), layer1_param['biases'])
    l1_output=tf.nn.relu(l1_output)
    
    l2_output= tf.add(tf.matmul(l1_output,layer2_param['weights']), layer2_param['biases'])
    l2_output=tf.nn.relu(l2_output)
    
    
    l3_output= tf.add(tf.matmul(l2_output,layer3_param['weights']), layer3_param['biases'])
    l3_output=tf.nn.relu(l3_output)
    
    l4_output= tf.add(tf.matmul(l3_output,layer4_param['weights']), layer4_param['biases'])
    l4_output=tf.nn.relu(l4_output)
    
    #The final output Layer
    output= tf.matmul(l4_output, output_layer_param['weights'])+output_layer_param['biases']
    
    return output # contains the output of the last output layer
    
    
"""
Up to this stage out network_model is ready, now we want to train it
"""


def train_model(x, epochs):
    
    # the previous function is called
    model_prediction= Network_model(x)
    
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=model_prediction, labels=y) )
    
    #using built in Adam Algorithm to minimize the cost                      
    optimizer=tf.train.AdamOptimizer().minimize(cost) 
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for i in range(epochs):
            loss=0
            for k in range(int(mnist.train.num_examples/batch_size)):
                xx,yy = mnist.train.next_batch(batch_size)
                l, c= sess.run([optimizer, cost], feed_dict={x:xx, y:yy})
                loss=loss+c 
            print "Epoch: ", i+1, "/", epochs, "  Loss is : ", loss    
            
        #Here I want to test the model on the test data    
        #staying in the same session, otherwise it will rise an error        
        evaluate= tf.equal(tf.argmax(model_prediction, 1), tf.arg_max(y, 1))
        accuracy=tf.reduce_mean(tf.cast(evaluate, 'float'))
        print "Accuracy is :", accuracy.eval({x:mnist.test.images, y:mnist.test.labels})
        
    
    
"""
Thanks God: Every thing is ready, now its time to train
"""   

DataPath="/tmp/data"
mnist= input_data.read_data_sets(DataPath, one_hot=True)

#Going to create a 4 Hidden layers model
no_neurons_layer1= 500
no_neurons_layer2= 300
no_neurons_layer3= 200
no_neurons_layer4=100

no_classes=10
batch_size=100
epochs=20


x= tf.placeholder('float',[None, 784] )
y= tf.placeholder('float')
"""
784 is due the size of the image 28x28 the mnist dataset contains images of size 28x28=784 pixels
these 784 pixels will be given as input to the model in flate array

"""
#Training Process starts
train_model(x, epochs)
    
"""
Note:
    To install tensorflow the following commands needs to be run in terminal 
    $ conda create -n tensorflow
    $ source activate tensorflow
    (tensorflow)$  # Your prompt should change
    
    # Linux/Mac OS X, Python 2.7/3.4/3.5, CPU only:
    (tensorflow)$ conda install -c conda-forge tensorflow
    
    or use
    $  conda install -c https://conda.anaconda.org/jjhelmus tensorflow    

"""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

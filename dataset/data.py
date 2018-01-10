import os 
import sys 
import time 


import numpy as np 
import tensorflow as tf 


# first step: create dataset 
# create iterator 
sess = tf.Session() 
dataset = tf.data.Dataset.range(100) 
iterator = dataset.make_one_shot_iterator()
next_ele = iterator.get_next() 

for i in range(100):
    value = sess.run(next_ele)
    print(value)
    assert i== value 


import os 
import sys 
import time 


import numpy as np 
import tensorflow as tf 

## Typical pipeline for reading data for training 
# The list of filenames
# Optional filename shuffling
# Optional epoch limit
# Filename queue
# A Reader for the file format
# A decoder for a record read by the reader
# Optional preprocessing
# Example queue
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


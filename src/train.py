""" Trains and evaluate posenet using a feed dictionary """ 

from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function 

import argparse 
import os 
import sys 
import time 

import tensorflow as tf 
import cv2 
import numpy as np 


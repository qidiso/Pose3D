from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import layers
from tensorflow.python.ops.array_ops import split
from tensorflow.python.keras._impl.keras.layers import Activation
from tensorflow.python.keras._impl.keras.layers import BatchNormalization
from tensorflow.python.keras._impl.keras.layers import Conv2D
from tensorflow.python.keras._impl.keras.layers import Conv2DTranspose
from tensorflow.python.keras._impl.keras.layers import Dense
from tensorflow.python.keras._impl.keras.layers import Input
from tensorflow.python.keras._impl.keras.layers import MaxPooling2D


def identity_block(input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.

        Arguments:
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        Returns:
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
                bn_axis = 3
        else:
                bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = Activation('relu', name=conv_name_base + '2a' + '_relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
        x = Activation('relu', name=conv_name_base + '2b' + '_relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

        x = layers.add([x, input_tensor], name='res' + str(stage) + block)
        x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
        return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        """conv_block is the block that has a conv layer at shortcut.

        Arguments:
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Tuple of integers.

        Returns:
            Output tensor for the block.

        Note that from stage 3, the first conv layer at main path is with
        strides=(2,2)
        And the shortcut should have strides=(2,2) as well
        """
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
                bn_axis = 3
        else:
                bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
        x = Activation('relu', name=conv_name_base + '2a' + '_relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
        x = Activation('relu', name=conv_name_base + '2b' + '_relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)

        x = layers.add([x, shortcut], name='res' + str(stage) + block)
        x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
        return x


def PoseNet(input_tensor=None, input_shape=None):
        """
        """

        # Determine proper input shape
        input_shape = (368, 368, 3)
        if input_tensor is None:
                img_input = Input(shape=input_shape)
        else:
                img_input = Input(tensor=input_tensor, shape=input_shape)

        if K.image_data_format() == 'channels_last':
                bn_axis = 3
        else:
                bn_axis = 1

        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
        x = Activation('relu', name='conv1_relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = conv_block(x, 3, [512, 512, 1024], stage=5, block='a')

        x = Conv2D(256, 1, name='res5b_branch2a_new')(x)
        x = Activation('relu', name='res5b_branch2a_relu')(x)

        x = Conv2D(128, 3, padding='same', name='res5b_branch2b_new')(x)
        x = Activation('relu', name='res5b_branch2b_relu')(x)

        x = Conv2D(256, 1, name='res5b_branch2c_new')(x)
        res_5b = Activation('relu', name='res5b_relu')(x)

        x = Conv2DTranspose(63, 4, strides=(2, 2), padding='same', use_bias=False, name='res5c_branch1a')(res_5b)
        delta_x, delta_y, delta_z = split(x, [21, 21, 21], axis=bn_axis, name='split_res5c_branch1a')
        sqr = layers.multiply([x, x], name='res5c_branch1a_sqr')
        x_sqr, y_sqr, z_sqr = split(sqr, [21, 21, 21], axis=bn_axis, name='split_res5c_branch1a_sqr')
        bone_length_sqr = layers.add([x_sqr, y_sqr, z_sqr], name='res5c_bone_length_sqr')
        res5c_bone_length = tf.pow(bone_length_sqr, 0.5, name='res5c_bone_length')

        x = Conv2DTranspose(128, 4, strides=(2, 2), padding='same', use_bias=False, name='res5c_branch2a')(res_5b)
        x = BatchNormalization(axis=bn_axis, name='bn5c_branch2a')(x)
        x = Activation('relu', name='res5c_branch2a_relu')(x)
        x = layers.concatenate([x, delta_x, delta_y, delta_z, res5c_bone_length], axis=bn_axis, name="res5c_branch2a_feat")
        x = Conv2D(128, 3, padding='same', name='res5c_branch2b')(x)
        x = Activation('relu', name='res5c_branch2b_relu')(x)
        x = Conv2D(84, 1, use_bias=False, name='res5c_branch2c')(x)
        heatmap, x_heatmap, y_heatmap, z_heatmap = split(x, [21, 21, 21, 21], axis=bn_axis, name='slice_heatmaps')

        return heatmap, x_heatmap, y_heatmap, z_heatmap


def main():
        temp = PoseNet()
        for item in temp:
                print("Operation:", item.shape)


if __name__ == '__main__':
        main()

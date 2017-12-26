# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image
Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import layers
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import decode_predictions  # pylint: disable=unused-import
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import preprocess_input  # pylint: disable=unused-import
from tensorflow.python.keras._impl.keras.engine.topology import get_source_inputs
from tensorflow.python.keras._impl.keras.layers import Activation
from tensorflow.python.keras._impl.keras.layers import AveragePooling2D
from tensorflow.python.keras._impl.keras.layers import BatchNormalization
from tensorflow.python.keras._impl.keras.layers import Conv2D
from tensorflow.python.keras._impl.keras.layers import Conv2DTranspose
from tensorflow.python.keras._impl.keras.layers import Dense
from tensorflow.python.keras._impl.keras.layers import Flatten
from tensorflow.python.keras._impl.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras._impl.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras._impl.keras.layers import Input
from tensorflow.python.keras._impl.keras.layers import MaxPooling2D
from tensorflow.python.keras._impl.keras.models import Model
from tensorflow.python.keras._impl.keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


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


def PoseNet(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000):
        """Instantiates the ResNet50 architecture.

  
        """

        # Determine proper input shape
        input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=197, data_format=K.image_data_format(), require_flatten=include_top, weights=weights)

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

        x = Conv2D(256, 1, name='res5b_branch2a')(x)
        x = Activation('relu', name='res5b_branch2a_relu')(x)

        x = Conv2D(128, 3, padding='same', name='res5b_branch2b')(x)
        x = Activation('relu', name='res5b_branch2b_relu')(x)

        x = Conv2D(256, 1, name='res5b_branch2c')(x)
        x = Activation('relu', name='res5b_branch2c_relu')(x)

        x = Conv2DTranspose(63, 4, strides=(2, 2), padding='same', use_bias=False, name='res5c_branch1a')(x)
        # TODO
        # res5c_brach1 层后面的层
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        if include_top:
                x = Flatten()(x)
                x = Dense(classes, activation='softmax', name='fc1000')(x)
        else:
                if pooling == 'avg':
                        x = GlobalAveragePooling2D()(x)
                elif pooling == 'max':
                        x = GlobalMaxPooling2D()(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
                inputs = get_source_inputs(input_tensor)
        else:
                inputs = img_input
        # Create model.
        model = Model(inputs, x, name='resnet50')

        return model


def main():
        temp = PoseNet()
        print(temp.name)
        for operation in temp.layers:
                print("Operation:", operation.name)


if __name__ == '__main__':
        main()

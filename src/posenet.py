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
import h5py
import numpy as np
from matplotlib import pyplot as plt
import cv2
from matplotlib.patches import Circle


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


def PoseNet(input_shape=(None, 368, 368, 3)):
        """
        """

        # Determine proper input shape
        img_input = tf.placeholder(tf.float32, shape=input_shape, name='img_input')

        if K.image_data_format() == 'channels_last':
                bn_axis = 3
        else:
                bn_axis = 1

        with tf.variable_scope("vnect"):
                net = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
                net = Activation('relu', name='conv1_relu')(net)
                net = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(net)

                net = conv_block(net, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
                net = identity_block(net, 3, [64, 64, 256], stage=2, block='b')
                net = identity_block(net, 3, [64, 64, 256], stage=2, block='c')

                net = conv_block(net, 3, [128, 128, 512], stage=3, block='a')
                net = identity_block(net, 3, [128, 128, 512], stage=3, block='b')
                net = identity_block(net, 3, [128, 128, 512], stage=3, block='c')
                net = identity_block(net, 3, [128, 128, 512], stage=3, block='d')

                net = conv_block(net, 3, [256, 256, 1024], stage=4, block='a')
                net = identity_block(net, 3, [256, 256, 1024], stage=4, block='b')
                net = identity_block(net, 3, [256, 256, 1024], stage=4, block='c')
                net = identity_block(net, 3, [256, 256, 1024], stage=4, block='d')
                net = identity_block(net, 3, [256, 256, 1024], stage=4, block='e')
                net = identity_block(net, 3, [256, 256, 1024], stage=4, block='f')

                net = conv_block(net, 3, [512, 512, 1024], stage=5, block='a')

                net = Conv2D(256, 1, name='res5b_branch2a_new')(net)
                net = Activation('relu', name='res5b_branch2a_relu')(net)

                net = Conv2D(128, 3, padding='same', name='res5b_branch2b_new')(net)
                net = Activation('relu', name='res5b_branch2b_relu')(net)

                net = Conv2D(256, 1, name='res5b_branch2c_new')(net)
                res_5b = Activation('relu', name='res5b_relu')(net)

                net = Conv2DTranspose(63, 4, strides=(2, 2), padding='same', use_bias=False, name='res5c_branch1a')(res_5b)
                delta_x, delta_y, delta_z = split(net, [21, 21, 21], axis=bn_axis, name='split_res5c_branch1a')
                sqr = layers.multiply([net, net], name='res5c_branch1a_sqr')
                x_sqr, y_sqr, z_sqr = split(sqr, [21, 21, 21], axis=bn_axis, name='split_res5c_branch1a_sqr')
                bone_length_sqr = layers.add([x_sqr, y_sqr, z_sqr], name='res5c_bone_length_sqr')
                res5c_bone_length = tf.pow(bone_length_sqr, 0.5, name='res5c_bone_length')

                net = Conv2DTranspose(128, 4, strides=(2, 2), padding='same', use_bias=False, name='res5c_branch2a')(res_5b)
                net = BatchNormalization(axis=bn_axis, name='bn5c_branch2a')(net)
                net = Activation('relu', name='res5c_branch2a_relu')(net)
                net = layers.concatenate([net, delta_x, delta_y, delta_z, res5c_bone_length], axis=bn_axis, name="res5c_branch2a_feat")
                net = Conv2D(128, 3, padding='same', name='res5c_branch2b')(net)
                net = Activation('relu', name='res5c_branch2b_relu')(net)
                net = Conv2D(84, 1, use_bias=False, name='res5c_branch2c')(net)
                heatmap, x_heatmap, y_heatmap, z_heatmap = split(net, [21, 21, 21, 21], axis=bn_axis, name='slice_heatmaps')
        return img_input, (heatmap, x_heatmap, y_heatmap, z_heatmap)


def load_weights(sess, model_file):
        f = h5py.File(model_file, 'r')
        params = tf.global_variables(scope="vnect")
        for v in params:
                [_, name, content] = v.name[:-2].split('/')
                print(name, content)
                if name[:2] == 'bn':
                        if content == 'gamma':
                                sess.run(v.assign(f['scale5c_branch2a']['weights'][:]))
                        elif content == 'beta':
                                sess.run(v.assign(f['scale5c_branch2a']['bias'][:]))
                        elif content == 'moving_mean':
                                sess.run(v.assign(f['bn5c_branch2a']['weights'][:]))
                        elif content == 'moving_variance':
                                sess.run(v.assign(f['bn5c_branch2a']['bias'][:]))
                else:
                        if name[:12] == 'res5a_branch':
                                name = name + '_new'
                        if content == 'kernel':
                                sess.run(v.assign(f[name]['weights'][:].transpose((2, 3, 1, 0))))
                        elif content == 'bias':
                                sess.run(v.assign(f[name]['bias'][:]))
        f.close()


def get_pose(img, heatmap, x_map, y_map, z_map):
        print("get pose")
        img_w = img.shape[1]
        img_h = img.shape[0]
        heat_w = heatmap.shape[2]
        heat_h = heatmap.shape[1]
        joints2d = np.ndarray(shape=[2, 21], dtype=float)
        joints3d = np.ndarray(shape=[3, 21], dtype=float)

        for i in range(21):
                hm = heatmap[0, :, :, i]
                ind = np.argmax(hm)
                w = ind % hm.shape[1]
                h = int(ind / hm.shape[1])
                w_ori = int(img_w * w / heat_w)
                h_ori = int(img_h * h / heat_h)
                joints2d[0, i] = h_ori
                joints2d[1, i] = w_ori

        # from IPython import embed; embed()
        return joints2d


def show_pose(img, joints2d):
        # fig, ax = plt.subplots(1)
        # ax.set_aspect('equal')
        # ax.imshow(img)
        # # for i in range(21):
        # 	circ = Circle((joints2d[0,i], joints2d[1,i]), 10)
        # 	ax.add_patch(circ)
        # # plt.show()
        for i in range(21):
                cv2.circle(img, (int(joints2d[0, i]), int(joints2d[1, i])), 10, (0, 0, 255), -1)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
        image_shape = (None, 540, 540, 3)
        img_file = '../dataset/mpii_3dhp_ts6/cam5_frame000160.jpg'
        img = cv2.imread(img_file)
        img = img[:540, 150:690]
        K.set_learning_phase(False)
        # img_input, (heatmap, x_heatmap, y_heatmap, z_heatmap) = PoseNet(input_shape=image_shape)
        img_input, output = PoseNet(input_shape=image_shape)

        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                load_weights(sess, 'vnect_model.h5')
                #out_img = sess.run(heatmap, feed_dict={img_input: [img]})
                out_ = sess.run(output, feed_dict={img_input: [img]})

        #print(out_img.shape, out_img.dtype)
        J2d = get_pose(img, out_[0], out_[1], out_[2], out_[3])
        show_pose(img, J2d)
        # plt.imshow(img)
        # plt.show()
        # for i in range(21):
        #         plt.imshow(out_img[0, :, :, i])
        #         plt.grid(True)
        #         plt.colorbar()
        #         plt.show()


if __name__ == '__main__':
        main()

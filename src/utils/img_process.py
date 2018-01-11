from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function 

import numpy as np 
import cv2 
import matplotlib.pyplot as plt 


def pad_image(img, target_size):
        t_h = target_size[0]
        t_w = target_size[1]
        h = img.shape[0]
        w = img.shape[1]
        if h == t_h and w == t_w:
                return img
        pad_h = int((t_h - h) / 2)
        pad_w = int((t_w - w) / 2)
        # from IPython import embed; embed()
        img_padded = np.ndarray([t_h, t_w, 3])
        for i in range(3):
                img_padded[:, :, i] = np.lib.pad(img[:, :, i], ((pad_h,), (pad_w,)), 'constant')

        return img_padded

# get center part of padded image given scale 
def resize_image(img, scale): 
        if scale == 1:
                return img 
        w = img.shape[1] 
        h = img.shape[0] 
        ch = img.shape[2] 
        virtual_w = int(w * scale) 
        virtual_h = int(h * scale) 
        virtual_img = cv2.resize(img, (virtual_w, virtual_h)) 
        # print(virtual_img.shape) 
        w_start = int(virtual_w/2 - w/2) 
        h_start = int(virtual_h/2 - h/2) 
        cen_img = virtual_img[h_start:h_start+h, w_start:w_start+w, :].copy() 
        return cen_img 

# get different scales of input image 
def preprocess_img(img, scales):
        img1 = np.swapaxes(img, 0, 1)
        img1 = img1 / 255 - 0.4
        img2 = cv2.resize(img1, (448, 848))
        img_stack = np.ndarray(shape=[848, 448, 3, 3])
        for i in range(3):
                # img_resized = img2
                # img_resized.resize([int(848 * scales[i]), int(448 * scales[i]), 3])
                img_resized = cv2.resize(img2, (int(448 * scales[i]), int(848 * scales[i])))
                img_pad = pad_image(img_resized, [848, 448])
                img_stack[:, :, :, i] = img_pad
        # from IPython import embed; embed()
        return img_stack


# compute final heatmap from network outputs 
# heatmap shape: [h, w, channel] 
def get_heatmap(output_stack, scales):
        hm_size = output_stack[0][0][0].shape 
        heatmap = np.zeros(hm_size) 
        xmap = np.zeros(hm_size) 
        ymap = np.zeros(hm_size) 
        zmap = np.zeros(hm_size) 
        # for i in range( len(scales) ): 
        for i in range(3): 
                raw_heatmap = output_stack[i][0][0] 
                raw_xmap = output_stack[i][1][0] 
                raw_ymap = output_stack[i][2][0] 
                raw_zmap = output_stack[i][3][0] 
                sized_heatmap = resize_image(raw_heatmap, 1 / scales[i]) 
                sized_xmap = resize_image(raw_xmap, 1 / scales[i]) 
                sized_ymap = resize_image(raw_ymap, 1 / scales[i]) 
                sized_zmap = resize_image(raw_zmap, 1 / scales[i]) 
                heatmap = heatmap + sized_heatmap 
                xmap = xmap + sized_xmap 
                ymap = ymap + sized_ymap 
                zmap = zmap + sized_zmap 
        factor = len(scales) 
        heatmap = heatmap / factor
        xmap = xmap / factor
        ymap = ymap / factor 
        zmap = zmap / factor 
        return heatmap, xmap, ymap, zmap # final heatmap 

# compute 2D and 3D pose from final heatmap 
# input heatmap shape: [h,w,channel] 
def get_pose(img, heatmap, xmap, ymap, zmap): 
        # transpose to be consistent with img's dimension order 
        heatmap = heatmap.swapaxes(0,1) 
        xmap = xmap.swapaxes(0,1) 
        ymap = ymap.swapaxes(0,1) 
        zmap = zmap.swapaxes(0,1) 
        # from IPython import embed; embed() 
        img_w = img.shape[1]
        img_h = img.shape[0]
        heat_w = heatmap.shape[1]
        heat_h = heatmap.shape[0]
        joints2d = np.ndarray(shape=[2, 21], dtype=float)
        joints3d = np.ndarray(shape=[3, 21], dtype=float)

        for i in range(21):
                hm = heatmap[:, :, i]
                xm = xmap[:,:,i] 
                ym = ymap[:,:,i] 
                zm = zmap[:,:,i] 
                ind = np.argmax(hm)
                w = ind % hm.shape[1]
                h = int(ind / hm.shape[1])
                w_ori = int(img_w * w / heat_w)
                h_ori = int(img_h * h / heat_h)
                joints2d[0, i] = h_ori
                joints2d[1, i] = w_ori
                x = 100 * xmap[h,w,i] 
                y = 100 * ymap[h,w,i] 
                z = 100 * zmap[h,w,i] 
                joints3d[0, i] = x
                joints3d[1, i] = y 
                joints3d[2, i] = z 

        # from IPython import embed; embed()
        return joints2d, joints3d 


def show_pose(img, joints2d):
        # fig, ax = plt.subplots(1)
        # ax.set_aspect('equal')
        # ax.imshow(img)
        # # for i in range(21):
        # 	circ = Circle((joints2d[0,i], joints2d[1,i]), 10)
        # 	ax.add_patch(circ)
        # # plt.show()
        jn = joints2d.shape[1] 
        for i in range(jn):
                cv2.circle(img, (int(joints2d[1, i]), int(joints2d[0, i])), 10, (0, 0, 255), -1)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

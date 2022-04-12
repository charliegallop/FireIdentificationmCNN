# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:48:29 2022

@author: cg639

following:https://towardsdatascience.com/data-preprocessing-and-network-building-in-cnn-15624ef3a28b
"""
#%%
import keras
import tensorflow
from skimage import io
import os
import glob
import numpy as np
import random
import matplotlib.pyplot as plt
import ipyplot

#%%

def plot_images(images, cmp = 'viridis'):
    numImages = len(images)
    f, ax = plt.subplots(1, numImages, sharey = True)
    f.set_figwidth(15)
    
    if len(images) > 1:
        for count, i in enumerate(images):
            if count > 0:
                ax[count].imshow(i, cmap = cmp)
                ax[count].set_title = f"{i}"
            else: 
                ax[count].imshow(i)
                ax[count].set_title = 'Original'
    else:
        ax.imshow(images[0])
    return f


#%%
# importing and Loading the data into the dataframe
#class 1 - fire, class 0 - no fire
DATASET_PATH = 'C:/Users/cg639/OneDrive - University of Exeter/CNNs/Dataset/Testing/fire'
fire_cls = ['fire', 'nofire']

#%%
# glob through the directory (returns a list of all file paths)
fire_path = '//tremictssan.fal.ac.uk/userdata/cg639/My Documents/CNNs/Dataset/Training and Validation/fire'
image_list_fire = os.listdir(fire_path)
#%%

# access some element (a file) from the list
image =io.imread(image_list_fire[0])

#%% Plotting the original image and the RGB Channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey = True)
f.set_figwidth(15)
ax1.imshow(image)

# RGB Channels
# ChannelID: 0 for Red, 1 for Green, 2 for Blue
ax2.imshow(image[:, :, 0]) # Red
ax3.imshow(image[:, :, 1]) # Green
ax4.imshow(image[:, :, 2]) # Blue
f.suptitle('Different Channels of Image')

#%% Image processing techniques

#1. Thresholding

# bin_image will be a (240, 320) True/False array
#The range of pixel varies between 0 and 255
#The pixel having black is more close to 0 and pixel
# which is white is more close to 255
# 125 is Arbitrary heuristic measure halfway between
# 1 and 255

bin_image = image[:, :, 0] > 125

plot_images([image, bin_image], "gray")
# f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)
# f.set_figwidth(15)
# ax1.imshow(image)
# ax2.imshow(bin_image, cmap = 'gray')

#%%
#2. Erosion, Dilation, Opening and Closing


# Erosion shrinks bright regions and enlarges dark regions. Dilation on the other hand is exact opposite side — it shrinks dark regions and enlarges the bright regions.
# Opening is erosion followed by dilation. Opening can remove small bright spots (i.e. “salt”) and connect small dark cracks. This tends to “open” up (dark) gaps between (bright) features.
# Closing is dilation followed by erosion. Closing can remove small dark spots (i.e. “pepper”) and connect small bright cracks. This tends to “close” up (dark) gaps between (bright) features.

from skimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening
from skimage.morphology import selem

# use a disk of radius 3

selem = selem.disk(3)

#opening and closing
open_img = binary_erosion(bin_image, selem)
close_img = binary_closing(bin_image, selem)

# erosion and dilation
eroded_img = binary_erosion(bin_image, selem)
dilated_img = binary_dilation(bin_image, selem)

plot_images([image, open_img, close_img, eroded_img, dilated_img], "gray")

# f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey = True)
# f.set_figwidth(15)
# ax1.imshow(bin_image, cmap = 'gray')
# ax2.imshow(open_img, cmap = 'gray')
# ax2.set_title("Open")
# ax3.imshow(close_img, cmap = 'gray')
# ax3.set_title("Close")
# ax4.imshow(eroded_img, cmap = 'gray')
# ax4.set_title("Eroded")
# ax5.imshow(dilated_img, cmap = 'gray')
# ax5.set_title("Dilated")

#%% Normalisation

# Important for helping with issue of propagating gradients

#way1-this is common technique followed in case of RGB images 
norm1_image = image/255
#way2-in case of medical Images/non natural images 
norm2_image = image - np.min(image)/np.max(image) - np.min(image)
#way3-in case of medical Images/non natural images 
norm3_image = image - np.percentile(image,5)/ np.percentile(image,95) - np.percentile(image,5)

plot_images([image, norm1_image, norm2_image, norm3_image])

# f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey = True)
# f.set_figwidth(15)
# ax1.imshow(image)
# ax2.imshow(norm1_image)
# ax2.set_title("Way #1")
# ax3.imshow(norm2_image)
# ax3.set_title("Way #2")
# ax4.imshow(norm3_image)
# ax4.set_title("Way #3")

#%% Augmentation

# Increasing the size of the dataset

from skimage import transform as tf

# flip left-right, up-down
image_flipr = np.fliplr(image)
image_flipud = np.flipud(image)

plot_images([image, image_flipr, image_flipud])

#%%

# Specify x and y coordinates to be used for shifting (mid points)
shift_x, shift_y, = image.shape[0]/2, image.shape[1]/2

# translation by certain units

matrix_to_topleft = tf.SimilarityTransform(translation = [-shift_x, - shift_y])
matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])

# Rotation

rot_transforms = tf.AffineTransform(rotation = np.deg2rad(45))
rot_matrix = matrix_to_topleft + rot_transforms + matrix_to_center
rot_image = tf.warp(image, rot_matrix)

# scaling 
scale_transforms = tf.AffineTransform(scale=(2, 2))
scale_matrix = matrix_to_topleft + scale_transforms + matrix_to_center
scale_image_zoom_out = tf.warp(image, scale_matrix)

scale_transforms = tf.AffineTransform(scale=(0.5, 0.5))
scale_matrix = matrix_to_topleft + scale_transforms + matrix_to_center
scale_image_zoom_in = tf.warp(image, scale_matrix)

# translation
transaltion_transforms = tf.AffineTransform(translation=(50, 50))
translated_image = tf.warp(image, transaltion_transforms)


plot_images([image, rot_image, scale_image_zoom_out, scale_image_zoom_in, translated_image])

#%%

# shear transforms
shear_transforms = tf.AffineTransform(shear=np.deg2rad(45))
shear_matrix = matrix_to_topleft + shear_transforms + matrix_to_center
shear_image = tf.warp(image, shear_matrix)

bright_jitter = image*0.999 + np.zeros_like(image)*0.001

plot_images([image, shear_image, bright_jitter])
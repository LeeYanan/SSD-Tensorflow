
# coding: utf-8

# In[1]:

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import os
import argparse


from datetime import datetime

slim = tf.contrib.slim

# In[2]:

#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import time
import subprocess


# In[3]:

import sys
sys.path.append('../')

precision = 10
def getCurrentClock():
    #return time.clock()
    return datetime.now()

# In[4]:

from nets import ssd_vgg_300, ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

procWidth = 1280 #640   # processing width (x resolution) of frame
procHeight = 720   # processing width (x resolution) of frame

# In[5]:

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


# ## SSD 300 Model
# 
# The SSD 300 network takes 300x300 image inputs. In order to feed any image, the latter is resize to this input shape (i.e.`Resize.WARP_RESIZE`). Note that even though it may change the ratio width / height, the SSD model performs well on resized images (and it is the default behaviour in the original Caffe implementation).
# 
# SSD anchors correspond to the default bounding boxes encoded in the network. The SSD net output provides offset on the coordinates and dimensions of these anchors.

# In[6]:
shapeWidth=512
shapeHeight=512
shapeWidth=300
shapeHeight=300

# Input placeholder.
#net_shape = (300, 300)
net_shape = (shapeWidth, shapeHeight)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")

args = vars(ap.parse_args())

data_format = 'NHWC' #'NHWC' #'NCHW'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
if shapeWidth==300:
    ssd_net = ssd_vgg_300.SSDNet()
else:
    ssd_net = ssd_vgg_512.SSDNet()

with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
if shapeWidth==300:
    ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
else:
    ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt.index'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# ## Post-processing pipeline
# 
# The SSD outputs need to be post-processed to provide proper detections. Namely, we follow these common steps:
# 
# * Select boxes above a classification threshold;
# * Clip boxes to the image shape;
# * Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
# * If necessary, resize bounding boxes to original image shape.

# In[7]:

# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(shapeWidth, shapeHeight)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# In[10]:


start_time = getCurrentClock()

#A smooth drive in The Crew on PS4 - OSSDC Simulator ACC Train 30fps
videoUrl = subprocess.Popen("youtube-dl -f22 -g https://www.youtube.com/watch?v=uuQlMCMT71I", shell=True, stdout=subprocess.PIPE).stdout.read()
videoUrl = videoUrl.decode("utf-8").rstrip()
print("videoUrl =",videoUrl)

# if the video argument is None, then we are reading from webcam
#videoUrl = args.get("video", None)

print("videoUrl=",videoUrl)

from imutils.video import WebcamVideoStream

webcam=False
#webcam=True

if webcam:
    #cap = WebcamVideoStream(src=""+str(videoUrl)+"").start()
    cap = WebcamVideoStream(videoUrl).start()
else:
    cap = cv2.VideoCapture(videoUrl)

#cap = cv2.VideoCapture(videoUrl)

count=50
#skip=2000
skip=0

SKIP_EVERY=150 #pick a frame every 5 seconds

count=1000000
#skip=0 #int(7622-5)
SKIP_EVERY=0

every=SKIP_EVERY
initial_time = getCurrentClock()
flag=True

frameCnt=0
prevFrameCnt=0
prevTime=getCurrentClock()

showImage=False
showImage=True
processImage=False
processImage=True
zoomImage=0
#zoomImage=True
rclasses = []
rscores = []
rbboxes = []

record = False
#record = True

out = None
if record:
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    out = cv2.VideoWriter('output-'+timestr+'.mp4',fourcc, 30.0, (int(procWidth),int(procHeight)))

#output_side_length = int(1920/zoomImage)

#height_offset = int((height - output_side_length) / 2)
#width_offset = int((width - output_side_length) / 2)

while True:
    #frame = cap.read()
    #if True:
    if webcam or cap.grab():
        if webcam:
            frame = cap.read()
        else:
            flag, frame = cap.retrieve()    
        if not flag:
            continue
        else:
            frameCnt=frameCnt+1
            nowMicro = getCurrentClock()
            delta = (nowMicro-prevTime).total_seconds()
            #print("%f " % (delta))
            if delta>=1.0:
                print("FPS = %0.4f" % ((frameCnt-prevFrameCnt)/delta))
                prevTime = nowMicro
                prevFrameCnt=frameCnt

            if skip>0:
                skip=skip-1
                continue
            
            if every>0:
                every=every-1
                continue
            every=SKIP_EVERY
            
            count=count-1
            if count==0:
                break

            img = frame
            if processImage:    
                if zoomImage>0:
                    #crop center of image, crop width is output_side_length
                    output_side_length = int(1920/zoomImage)
                    height, width, depth = frame.shape
                    #print (height, width, depth)
                    height_offset = int((height - output_side_length) / 2)
                    width_offset = int((width - output_side_length) / 2)
                    #print (height, width, depth, height_offset,width_offset,output_side_length)
                    img = frame[height_offset:height_offset + output_side_length,width_offset:width_offset + output_side_length]
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                start_time = getCurrentClock()
                rclasses, rscores, rbboxes =  process_image(img)
                if len(rclasses)>0:
                    nowMicro = getCurrentClock()
                    print("# %s - %s - %0.4f seconds ---" % (frameCnt,rclasses.astype('|S3'), (nowMicro - start_time).total_seconds()))
                    start_time = nowMicro
                if showImage:
                    visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
            if showImage:
                #visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
                #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
                if processImage:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow("ssd",img)
            if record:
                #if processImage:
                    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                newimage = cv2.resize(img,(procWidth,procHeight))
                out.write(newimage)
    key = cv2.waitKey(1)
    if  key == 27:
        break
    elif key == ord('u'):
        showImage= not(showImage)
    elif key == ord('p'):
        processImage= not(processImage)
    elif key == ord('z'):
        zoomImage=zoomImage+1
        if zoomImage==10:
            zoomImage=0
    elif key == ord('x'):
        zoomImage=zoomImage-1
        if zoomImage<0:
            zoomImage=0

nowMicro = getCurrentClock()
print("# %s -- %0.4f seconds - FPS: %0.4f ---" % (frameCnt, (nowMicro - initial_time).total_seconds(), frameCnt/(nowMicro - initial_time).total_seconds()))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




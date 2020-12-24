import matplotlib.pyplot as plt
import numpy as np
import json
import os
import cv2
from pycocotools.coco import COCO
import skimage.io as io
import random
import tensorflow as tf
import matplotlib.gridspec as gridspec

dataDir = 'F:\datasets\COCO 2017'
dataType = 'val'

def get_coco(dataDir, dataType):
  annFile = '{}/annotations/instances_{}2017.json'.format(dataDir, dataType)

  # initialize the COCO api for instance annotations
  coco = COCO(annFile)
  return coco


def get_mask(info, coco):

    annIds = coco.getAnnIds(imgIds=info['id'])
    anns = coco.loadAnns(annIds)
    mask = np.zeros((info['height'], info['width']))
    for i in range(len(anns)):
        mask = np.maximum(coco.annToMask(anns[i])*(i+1), mask)

    return mask


def transform(boxes):

    xmin, ymin, w, h = [boxes[:, i] for i in range(4)]

    xmax = xmin + w
    ymax = ymin + h
    boxes = np.stack([xmin, ymin, xmax, ymax], axis=-1)
    return boxes
def get_bb_box(info,coco):
    annIds = coco.getAnnIds(imgIds=info['id'])
    anns = coco.loadAnns(annIds)
    boxes = []
    for ann in anns:
        bbox = ann['bbox']
        boxes.append(bbox)

    boxes = np.array(boxes)
    boxes = transform(boxes)
    return boxes


def get_proposals(img_size):
    scales = [0.5,0.9]
    asp_r  = [1/2,1,2]
    xmin   = np.arange(0,img_size[1],16)
    ymin   = np.arange(0,img_size[0],16)
    priors = np.zeros(shape = (xmin.shape[0],ymin.shape[0],1))
    x      = np.repeat(np.expand_dims(xmin, -1), ymin.shape, -1)
    y      = np.repeat(np.expand_dims(ymin, 0), xmin.shape, 0)

    for scale in scales:
        xy = np.stack([x, y], axis=-1)
        wh = np.zeros_like(xy)

        for ratio in asp_r:
            wh[:,:,0] = (scale/np.sqrt(ratio))*img_size[1]
            wh[:,:,1] = scale * np.sqrt(ratio)*img_size[0]
            priors    = np.concatenate([priors,xy, wh],axis = -1)

    return  priors[:,:,1:]

def rearrange(proposals):
    no_boxes = proposals.shape[-1]//4

    boxes = np.zeros(shape = (1,4))
    for i in range(no_boxes):
        box   = proposals[:,:,i*4:(i+1)*4]
        box   = box.reshape(-1,4)
        boxes = np.concatenate([boxes,box], axis = 0)
    return boxes[1:]

def get_iou(box1,box2):
    if len(box1.shape) == 2:
        xmin1, ymin1, xmax1, ymax1 = [box1[:,i] for i in range(4)]
    else:
        xmin1, ymin1, xmax1, ymax1 = [box1[i] for i in range(4)]

    xmin2, ymin2, xmax2, ymax2 = [box2[:,i] for i in range(4)]

    xmin = np.maximum(xmin1, xmin2)
    ymin = np.maximum(ymin1, ymin2)
    xmax = np.minimum(xmax1, xmax2)
    ymax = np.minimum(ymax1, ymax2)


    inter_area = np.maximum(ymax - ymin,0) * np.maximum(xmax - xmin,0)

    area1      = (xmax1 - xmin1)*(ymax1 - ymin1)
    area2      = (xmax2 - xmin2) * (ymax2 - ymin2)

    iou        = inter_area/(area1 + area2 - inter_area)

    return iou,area1,area2


def filter(boxes, img_size):
    iou, area1, area2 = get_iou(np.array((0, 0, img_size[0], img_size[1])), boxes)
    index             = np.logical_and(iou > 0.3*(area2/area1),iou <= area2/area1)
    boxes             = boxes[index]
    x_boxes           = np.stack([boxes[:, 0],boxes[:, 2]], axis = -1)
    y_boxes           = np.stack([boxes[:, 1],boxes[:, 3]], axis = -1)

    x_boxes           = np.clip(x_boxes, 0., img_size[1]-1)
    y_boxes           = np.clip(y_boxes, 0., img_size[0]-1)
    print(x_boxes.max(),y_boxes.max())
    boxes             = np.stack([x_boxes[:,0],y_boxes[:,0],x_boxes[:,1],y_boxes[:,1]],axis = -1)

    return boxes

def get_center(boxes):
    xmin, ymin, xmax, ymax = [boxes[:, i] for i in range(4)]
    x_center = (xmin + xmax)//2
    y_center =
def get_truth_index(boxes, proposals):

    for box in boxes:
        iou,area1,area2 =









coco = get_coco(dataDir, dataType)
info = coco.dataset['images'][0]
annIds = coco.getAnnIds(imgIds=info['id'])
anns = coco.loadAnns(annIds)
boxes = get_bb_box(info,coco)

file_path = 'F:\datasets\COCO 2017\\val2017'
img = plt.imread(os.path.join(file_path,info['file_name']))
img_size = img.shape[:-1]
proposals = get_proposals(img.shape[:-1])
boxes                = rearrange(proposals)
boxes                = transform(boxes)
boxes                = filter(boxes,img_size)

print(boxes.shape)
xmin,ymin,xmax,ymax = [boxes[:,i] for i in range(4)]


xmin,xmax            = xmin/info['width'] , xmax/info['width']
ymin,ymax            = ymin/info['height'] , ymax/info['height']

boxes                = np.stack([ymin,xmin,ymax,xmax],axis = -1)

img                  = tf.image.draw_bounding_boxes([img/255.],[[boxes]],colors=[[1.0,0.,0.]])

print(boxes.shape)
plt.imshow(img[0])
plt.show()
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import json
from PIL import Image, ImageDraw


def get_data(Datadir,datatype):
    Datadir = Datadir.decode('UTF-8')
    datatype = datatype.decode('UTF-8')
    annFile = '{}/annotations/instances_{}2017.json'.format(Datadir, datatype)
    data = open(annFile)
    data = json.load(data)
    return data


def get_annotations(info, data):
    anns = []
    for ann in data['annotations']:
        if ann['image_id'] == info['id']:
            anns.append(ann)

    return anns

def get_mask(coords,img_size):
    image = Image.new("1", img_size)
    draw = ImageDraw.Draw(image)
    for coord in coords:
        draw.polygon(coord, fill=200)

    mask = np.array(image)

    return mask

def transform(boxes):
    xmin, ymin, w, h = [boxes[:, i] for i in range(4)]
    xmax = xmin + w
    ymax = ymin + h
    boxes = np.stack([xmin, ymin, xmax, ymax], axis=-1)
    return boxes


def get_bb_box_and_segmentation(anns):
    boxes = []
    segmentation = []
    for ann in anns:
        if ann['iscrowd'] == 1:
            continue
        boxes.append(ann['bbox'])
        segmentation.append(ann['segmentation'])
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return None,None
    boxes = transform(boxes)
    return boxes,segmentation


def get_proposals(img_size):
    scales = [0.5,0.9]
    asp_r  = [1/2,1,2]
    xmin   = np.arange(0,img_size[0],16)
    ymin   = np.arange(0,img_size[1],16)

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
    iou, area1, area2 = get_iou(np.array((0, 0, img_size[1], img_size[0])), boxes)
    index             = np.logical_and(iou > 0.3*(area2/area1),iou <= area2/area1)
    boxes             = boxes[index]
    x_boxes           = np.stack([boxes[:, 0],boxes[:, 2]], axis = -1)
    y_boxes           = np.stack([boxes[:, 1],boxes[:, 3]], axis = -1)

    x_boxes           = np.clip(x_boxes, 0., img_size[1]-1)
    y_boxes           = np.clip(y_boxes, 0., img_size[0]-1)
    boxes             = np.stack([x_boxes[:,0],y_boxes[:,0],x_boxes[:,1],y_boxes[:,1]],axis = -1)

    return boxes

def get_center(boxes):
    xmin, ymin, xmax, ymax = [boxes[:, i] for i in range(4)]
    x_center = (xmin + xmax)//2
    y_center = (ymin + ymax)//2

    return np.stack([x_center,y_center], axis = -1)
def get_truth_index(boxes, proposals):
    proposals_center = get_center(proposals)
    box_centers      = get_center(boxes)
    truth_index      = np.zeros(shape = (len(proposals)), dtype = np.bool)
    instance_no      = np.zeros(shape=(len(proposals)), dtype=np.uint8)
    i = 0
    for box,center in zip(boxes,box_centers):
        iou,area1,area2 = get_iou(box,proposals)

        index              = iou == area1/area2

        diff               = center - proposals_center
        error              = np.sqrt(np.sum(np.square(diff), axis = -1))
        index              = np.logical_and(index,error <= 16.)
        instance_no[index] = i
        truth_index        = np.logical_or(truth_index,index)
        i                  += 1
    return truth_index,instance_no

def get_ground_truth(coordinates,img_size,segmentation,img,output_size,filename):
    proposals = get_proposals(img_size)
    proposals = rearrange(proposals)
    proposals = transform(proposals)
    proposals = filter(proposals, img_size)
    truth_index,instance_nos = get_truth_index(coordinates, proposals)
    patches,masked_patches,y = get_ground_truth_mask(img,truth_index,proposals,segmentation,instance_nos,output_size,img_size,filename)
    return patches,masked_patches,y

def get_ground_truth_mask(img,truth_index,proposals,segmentation,instance_nos,output_size,img_size,filename):
    boxes = proposals[truth_index]
    instance_nos = instance_nos[truth_index]
    patches,masked_patches = [],[]
    y = []

    for i,box in zip(instance_nos,boxes):
        box   = np.int32(box)
        image = img[box[1]:box[3], box[0]:box[2]]
        image = tf.image.resize(image,size=output_size)
        patches.append(image)
        try:
            mask = get_mask(segmentation[i],img_size)
        except:
            print(i,filename)
        instance = np.where(mask, 1., -1.)
        masked_patch = instance[box[1]:box[3], box[0]:box[2]]
        masked_patch = tf.image.resize(tf.expand_dims(masked_patch,-1),size=output_size)
        masked_patch = masked_patch[:,:,0]
        masked_patches.append(masked_patch)
        y.append(1.0)
    remaining_boxes = max(len(masked_patches),2)
    negative_boxes  = proposals[np.logical_not(truth_index)]
    for i in range(remaining_boxes):
        box = np.int32(negative_boxes[i])

        image = img[box[1]:box[3], box[0]:box[2]]
        image = tf.image.resize(image,size=output_size)
        patches.append(image)

        masked_patches.append(-np.ones(shape = output_size,dtype = np.int32))
        y.append(-1.)
    return np.array(patches),np.array(masked_patches),np.array(y)


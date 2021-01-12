import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pycocotools.coco import COCO
import pandas as pd
from objectpath import *

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

def save_masks(datatype,datadir,save_dir):
    coco = get_coco(datadir,datatype)

    for i,info in enumerate(coco.dataset['images']):
        filename = info['file_name'].split('.')[0]
        mask = get_mask(info,coco)
        plt.imsave(os.path.join(save_dir,filename+'.png'),mask,cmap = 'gray')
        if i%10000 == 0:
            print(i)

dataDir = 'F:\datasets\COCO 2017'
dataType = 'train'


def get_csv(datatype,datadir):
    obj = open('{}/annotations/instances_{}2017.json'.format(datadir, datatype))
    data = json.load(obj)
    anns = data['annotations']
    df = {'image_id': [], 'xmin': [], 'ymin': [], 'width': [], 'height': []}
    for i,ann in enumerate(anns):
        xmin, ymin, w, h = ann['bbox']
        id = ann['image_id']
        df['image_id'].append(id)
        df['xmin'].append(xmin)
        df['ymin'].append(ymin)
        df['width'].append(w)
        df['height'].append(h)
        if i%10000 == 0:
            print(i,'  first loop')

    df = pd.DataFrame(data=df)
    df['file_name'] = '0'
    df['image_height'] = 0
    df['image_width'] = 0

    infos = data['images']
    for i,info in enumerate(infos):
        id = info['id']
        indices = np.argwhere((df['image_id'] == id).values)
        df.loc[indices[:, 0], 'file_name'] = info['file_name']
        df.loc[indices[:, 0], 'image_height'] = info['height']
        df.loc[indices[:, 0], 'image_width'] = info['width']
        if i % 1000 == 0:
            print(i,' Second')

    return df

Datadir = 'F:/datasets/COCO 2017'
datatype = 'train'
save_masks(datatype,Datadir,'F:\datasets\COCO 2017\\train mask2017')

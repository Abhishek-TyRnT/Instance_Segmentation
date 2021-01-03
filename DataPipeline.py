import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.image import rot90
from matplotlib.pyplot import imread
import os
from Utils import get_coco,get_ground_truth,get_bb_box,get_mask
import time


def data_generator(imgdir,Datadir,datatype,output_size,img_size, use_aug = True):
    imgdir = imgdir.decode('UTF-8')
    coco = get_coco(Datadir,datatype)
    imgdir = os.path.join(imgdir,datatype.decode('UTF-8') + '2017')


    infos = coco.dataset['images']

    for info in infos:
        boxes    = get_bb_box(info, coco)
        if boxes is None:
            continue
        mask     = get_mask(info, coco)
        img      = imread(os.path.join(imgdir,info['file_name']))
        img_size = img.shape[:-1]
        patches,masked_patches,label  = get_ground_truth(boxes,img_size,mask,img,output_size)
        shuffle = np.random.permutation(len(patches))
        patches = patches[shuffle]
        masked_patches = masked_patches[shuffle]
        label          = label[shuffle]


        for patch,masked_patch,y in zip(patches,masked_patches,label):

            yield patch/255.0,masked_patch,y

            if use_aug:
                x = np.random.choice([0,1],size = 1, replace = False, p = [0.8,0.2])
                if x == 1:
                    rotation = np.random.randint(1,4,size = 1)
                    patch    = rot90(patch,rotation[0])
                    masked_patch = rot90(np.expand_dims(masked_patch,-1),rotation[0])
                    masked_patch = masked_patch[:,:,0]

                    yield patch/255.0,masked_patch,y


def get_dataset(img_dir,Datadir,datatype,output_size,img_size, use_aug = True):
    args = [
        img_dir,
        Datadir,
        datatype,
        output_size,
        img_size,
        use_aug
    ]
    dtypes = (tf.float32, tf.float32, tf.float32)
    shapes = (tf.TensorShape((output_size[0],output_size[1],3)),
              tf.TensorShape((output_size[0],output_size[1])),
              tf.TensorShape(()))
    return Dataset.from_generator(data_generator,output_types = dtypes, output_shapes= shapes, args = args)

"""input_shape = (224,224,3)
output_shape = (224,224)
file_dir   = 'F:/datasets/COCO 2017'
train_dataset = get_dataset(file_dir,file_dir,'train',output_shape,input_shape)
i = time.time()
for x in train_dataset.batch(5).take(5):
    print('hi')
print(time.time() - i)"""
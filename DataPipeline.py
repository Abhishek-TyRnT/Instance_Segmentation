import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.image import rot90
from matplotlib.pyplot import imread
import os
from Utils import get_ground_truth,get_bb_box_and_segmentation,get_data,get_annotations


def data_generator(Datadir,datatype,output_size, use_aug = True):
    data = get_data(Datadir,datatype)
    imgdir = os.path.join(Datadir.decode('UTF-8'),datatype.decode('UTF-8') + '2017')

    infos   = data['images']
    infos   = np.array(infos)
    shuffle = np.random.permutation(len(infos))
    infos   = infos[shuffle]

    for info in infos:
        anns = get_annotations(info, data)
        boxes,segmentation    = get_bb_box_and_segmentation(anns)

        if boxes is None:
            continue
        img      = imread(os.path.join(imgdir,info['file_name']))
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img,-1),3,-1)
        img_size = (info['width'], info['height'])

        patches,masked_patches,label  = get_ground_truth(boxes,img_size,segmentation,img,output_size,info['file_name'])
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


def get_dataset(datadir,datatype,output_size, use_aug = True):

    args = [
        datadir,
        datatype,
        output_size,
        use_aug
    ]
    dtypes = (tf.float32, tf.float32, tf.float32)
    shapes = (tf.TensorShape((output_size[0],output_size[1],3)),
              tf.TensorShape((output_size[0],output_size[1])),
              tf.TensorShape(()))
    return Dataset.from_generator(data_generator,output_types=dtypes, output_shapes= shapes, args = args).\
        apply(tf.data.experimental.prefetch_to_device("/gpu:0"))







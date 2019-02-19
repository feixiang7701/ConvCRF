"""
This is just a test program,the Layer variable is only be initialized
and its parameters is set freestyle. 
TODO:
[1] A real iamge segmentation project with ConvCRF.
"""


import tensorflow as tf
tf.enable_eager_execution()



image_name="./data/0cdf5b5d0ce1_01.jpg"
label_name="./data/0cdf5b5d0ce1_01_mask.gif"

def _process_pathnames(fname, label_path):
    img_str = tf.read_file(fname)
    img = tf.image.decode_jpeg(img_str)

    label_str = tf.read_file(label_path)
    # These are gif images so they return as (num_frames, h, w, c)
    label_img = tf.image.decode_gif(label_str)[0]

    label_img = label_img[:, :, 0] 

    label_car=tf.equal(label_img,255)
    label_car=tf.to_float(label_car)
    label_background=tf.not_equal(label_img,255)
    label_background=tf.to_float(label_background)

    label_car = tf.expand_dims(label_car, axis=-1) 
    label_background=tf.expand_dims(label_background,axis=-1)
    label_img = tf.concat([label_car, label_background], axis=-1)
    return img, label_img

img,label_img=_process_pathnames(image_name,label_name)
img=tf.expand_dims(img,axis=0)
label_img=tf.expand_dims(label_img,axis=0)

#ConvCRF Test
from ConvCrf_layer import ConvCRF
image_dims=(1280,1918)
CRF=ConvCRF(image_dims,7,4,2,160,10,10,10)
outputs=CRF([label_img,img])
results=tf.argmax(outputs,axis=3)
results=results.numpy()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
plt.imshow(results[0])
plt.title("Masked Image After ConvCRF")
plt.show()

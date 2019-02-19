

import tensorflow as tf
from tensorflow.python.keras.layers import  Layer
import numpy as np


def _diagonal_initializer(shape,dtype=tf.float32,partition_info=None):
    return tf.eye(shape[0], shape[1], dtype=tf.float32)


def _potts_model_initializer(shape,dtype=tf.float32,partition_info=None):
    return -1 * _diagonal_initializer(shape,dtype,partition_info)

def _get_ind(dz):
    if dz == 0:
        return 0, 0
    if dz < 0:
        return 0, -dz
    if dz > 0:
        return dz, 0


def _negative(dz):
    """
    Computes -dz for numpy indexing. Goal is to use as in array[i:-dz].

    However, if dz=0 this indexing does not work.
    None needs to be used instead.
    """
    if dz == 0:
        return None
    else:
        return -dz



class ConvCRF(Layer):
    """
    Implement the ConvLayer described in:
    Convolutional CRFs for Semantic Segmentation
    Marvin T. T. Teichmann, Roberto Cipolla
    """
    def __init__(self, image_dims, filter_size,blur,num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations, **kwargs):
        self.image_dims = image_dims
        self.filter_size=filter_size
        self.blur=blur
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.gauss_ker_weight_train = False
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        self.gauss_ker_weights=None
        super(ConvCRF, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weights of the spatial kernel
        self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                   shape=(self.num_classes, self.num_classes),
                                                   dtype=tf.float32,
                                                   initializer=_diagonal_initializer,
                                                   trainable=True)

        # Weights of the bilateral kernel
        self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                     shape=(self.num_classes, self.num_classes),
                                                     dtype=tf.float32,
                                                     initializer=_diagonal_initializer,
                                                     trainable=True)

        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_classes, self.num_classes),
                                                    dtype=tf.float32,
                                                    initializer=_potts_model_initializer,
                                                    trainable=True)

        super(ConvCRF, self).build(input_shape)

    def _create_mesh(self, requires_grad=False):
        cord_range = [range(s) for s in self.image_dims]
        mesh = np.array(np.meshgrid(*cord_range, indexing='ij'),dtype=np.float32)
        return mesh


    def _pos_feature(self,mesh,sdim,bs=1):
        #[1,2,h,w]
        return tf.stack(bs*[mesh*(1.0/sdim)])

    def _color_feature(self,rgb):
        #rgb NCHW
        return tf.stack(rgb*(1.0/self.theta_alpha))

    def pos_feature(self,bs=1):
        mesh=self._create_mesh()
        return self._pos_feature(mesh,self.theta_gamma,bs)

    def color_pos_feature(self,rgb,bs=1):
        rgb_norm=self._color_feature(rgb)
        mesh=self._create_mesh()
        pos_norm=self._pos_feature(mesh,self.theta_beta,bs)
        return tf.concat([rgb_norm,pos_norm],axis=1)

    def _create_convolutional_filters(self,features,bs):

        #features NCHW
        span=self.filter_size//2
        if self.blur>1:
            features=tf.keras.layers.AveragePooling2D((self.blur,self.blur),(self.blur,self.blur),padding='same',
                                                      data_format="channels_first")(features)
            
        h=features.shape[2]
        w=features.shape[3]

        gaussian_filter_np=np.zeros([bs, self.filter_size, self.filter_size, h, w],dtype=np.float32)
        
        for dx in range(-span,span+1):
            for dy in range(-span,span+1):
                
                dx1,dx2=_get_ind(dx)
                dy1,dy2=_get_ind(dy)

                feature_1=features[:,:,dx1:_negative(dx2),dy1:_negative(dy2)]
                feature_2=features[:,:,dx2:_negative(dx1),dy2:_negative(dy1)]

                diff_sq=(feature_1-feature_2)*(feature_1-feature_2)

                diff_exp=tf.exp(tf.reduce_sum(-0.5*diff_sq,axis=1))
                gaussian_filter_np[:,dx+span,dy+span,dx2:_negative(dx1),dy2:_negative(dy1)]=diff_exp.numpy()

        gaussian_filter = tf.get_variable("my_non_trainable",
                                          dtype=tf.float32,
                                          initializer=tf.constant(gaussian_filter_np),
                                          trainable=False)
        gaussian_filter_np=0

        return  tf.reshape(gaussian_filter,(bs,1,self.filter_size*self.filter_size,h,w))


    def compute_gaussian(self,input,filter):


       #(1)input im2col,shape(bs,c,self.filter_size,self.filter_size,h,w)
       #blur used to expand the receptive field
        if self.blur>1:
            input=tf.keras.layers.AveragePooling2D((self.blur,self.blur),(self.blur,self.blur),
                                                   padding="SAME")(input)

        bs, h, w, c = input.shape
        #im2col input-> (bs,c,self.filter_size,self.filter_size,h,w) im2col

        #(bs,h,w,self.filter_size*filter_size*c)
        input_col=tf.extract_image_patches(input,[1,self.filter_size,self.filter_size,1],[1,1,1,1],[1,1,1,1],padding='SAME')

        #(bs,c,self.filter_size*filter_size,h,w)
        input_col =tf.reshape(input_col,(bs,h,w,c,self.filter_size*self.filter_size))
        input_col =tf.transpose(input_col,perm=[0,3,4,1,2])
        #features*gauss_filter(bs,c,self.filter_size,self.filter_size,h,w)
        product=input_col*filter
        product=tf.reduce_sum(product,axis=2)

        return product



    def call(self,input):
        #NCHW tensorflo im2col
        #unary=tf.transpose(input[0][:,:,:,:],perm=(0,3,1,2))
        rgb=tf.transpose(input[1][:,:,:,:],perm=[0,3,1,2])

        bs, h, w, c = input[0].shape


        #Spatial filtering
        pos_feature = self.pos_feature(bs)
        spatial_filter = self._create_convolutional_filters(pos_feature, bs)

        #Bilateral filtering((pi,pi),(Ii,Ij)),self.filter_size
        rgb = np.array(rgb, dtype=np.float32)
        color_pos_feature = self.color_pos_feature(rgb, bs)
        bilateral_filter = self._create_convolutional_filters(color_pos_feature, bs)


        #ensure the input from the last layer
        all_ones = tf.ones((bs,h,w,c), dtype=np.float32)

        #norm
        spatial_norm=self.compute_gaussian(all_ones,spatial_filter)
        bilateral_norm=self.compute_gaussian(all_ones,bilateral_filter)


        for i in range(self.num_iterations):

            # Spatial filtering
            spatial_out = self.compute_gaussian(all_ones,spatial_filter)
            spatial_out = spatial_out / (spatial_norm+1e-20)

            # Bilateral filtering
            bilateral_out =self.compute_gaussian(all_ones,spatial_filter)
            bilateral_out = bilateral_out / (bilateral_norm+1e-20)

            # Weighting filter outputs

            spatial_ker_weights=tf.stack(bs*[self.spatial_ker_weights])
            bilateral_ker_weights=tf.stack(bs*[self.bilateral_ker_weights])
            message_passing = (tf.matmul(spatial_ker_weights,
                                         tf.reshape(spatial_out, (bs,c, -1))) +
                               tf.matmul(bilateral_ker_weights,
                                         tf.reshape(bilateral_out, (bs,c, -1))))

            # Compatibility transform
            compatibility_matrix=tf.stack(bs*[self.compatibility_matrix])
            pairwise = tf.matmul(compatibility_matrix, message_passing)

            # Adding unary potentials,h,w
            if self.blur>1:
                h_b=spatial_out.shape[2]
                w_b=spatial_out.shape[3]
                pairwise = tf.reshape(pairwise, (bs, c, h_b, w_b))
                pairwise = tf.transpose(pairwise,perm=(0,2,3,1))
                pairwise=tf.image.resize_bilinear(pairwise,(h,w))


            q_values = input[0] - pairwise
        return q_values

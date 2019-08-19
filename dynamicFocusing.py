"""
The official Tensorflow implementation of the dynamic focusing layer
for US RX beamforming proposed in:

"Learning beamforming in ultrasound imaging", Proc. MIDL 2019.

Some of the code is based on the official implementation
of the following paper:
Jaderberg et al., Spatial Transformer Networks, NIPS 2015.

"""

import tensorflow as tf
from scipy.io import loadmat
import numpy as np
import os
layers = tf.layers
os.environ["CUDA_VISIBLE_DEVICES"]="1"



def dyanamic_focusing_layer(input_fmap, Theta, specs, elemCoor, trainable=False):

    dims = input_fmap._shape_as_list()
    theta_init = tf.constant(Theta)
    theta = tf.Variable(initial_value=theta_init, expected_shape=Theta.shape[-1], trainable=trainable)

    c = tf.constant(np.squeeze(specs['SpeedOfSound']).astype(np.float32))
    fs = tf.constant(np.array(specs['IQSampleRate']).astype(np.float32))
    t = tf.constant(np.arange(dims[1], dtype=np.float32)) / fs
    w0 = tf.constant(np.array(2.0 * np.pi * specs['DemodulationFrequency']).astype(np.float32))
    ee, tt, ll = tf.meshgrid(elemCoor,t,theta)
    r = 0.5 * tf.multiply(tt ,c)
    x_rx = tf.multiply(r,tf.sin(ll))
    z_rx = tf.multiply(r,tf.cos(ll))
    delays_grid_t = (r+(tf.sqrt(tf.square(x_rx-ee)+tf.square(z_rx))))/c
    delays_grid = delays_grid_t*fs
    delays_grid = tf.clip_by_value(delays_grid,clip_value_min=0.0, clip_value_max=dims[1]-1.0)

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, delays_grid)
    cos_phi = tf.expand_dims(tf.cos(w0*(delays_grid_t-tt)),axis=3)
    sin_phi = tf.expand_dims(tf.sin(w0*(delays_grid_t-tt)),axis=3)
    real,imag = tf.split(out_fmap,num_or_size_splits=2,axis=0)
    IQx = real*cos_phi-imag*sin_phi
    IQy = real*sin_phi+imag*cos_phi
    out = tf.concat([IQx,IQy],axis=0)
    return out


def get_pixel_value(img, h,w,d):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = img._shape_as_list()
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth = shape[3]
    channels = shape[4]

    h = tf.expand_dims(h,4)
    w = tf.expand_dims(w,4)
    d = tf.expand_dims(d,4)
    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1, 1, 1))
    b = tf.tile(batch_idx, (1, height, width, depth, channels))

    h = tf.tile(h,(batch_size, 1, 1, 1, 1))
    w = tf.tile(w, (batch_size, 1, 1, 1, 1))
    d = tf.tile(d, (batch_size, 1, 1, 1, 1))
    indices = tf.stack([b, h, w, d, tf.zeros(shape=b._shape_as_list(),dtype=tf.int32)], 5)
    return tf.gather_nd(img, indices)


def bilinear_sampler(img, delays_grid):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    shape = img._shape_as_list()
    H = shape[1]
    W = shape[2]
    D = shape[3]
    max_d = tf.cast(D, 'int32')
    max_w = tf.cast(W, 'int32')
    d = tf.range(0,max_d)
    w = tf.range(0,max_w)

    # grab 4 nearest points to delays points
    w = tf.reshape(w,[1,W,1])
    w = tf.tile(w,[H,1,D])
    w = tf.expand_dims(w,0)
    h0 = tf.cast(tf.floor(delays_grid), 'int32')
    h0 = tf.expand_dims(h0,0)
    h1 = h0 + 1
    d = tf.reshape(d,[1,1,D])
    d = tf.tile(d,[H,W,1])
    d = tf.expand_dims(d,0)

    # get pixel value at NN coords
    Ia = get_pixel_value(img,h0,w,d)
    Ib = get_pixel_value(img,h1, w, d)

    # recast as float for delta calculation
    h0 = tf.cast(h0, 'float32')
    h1 = tf.cast(h1, 'float32')

    # calculate deltas
    wa = h1 - delays_grid
    wb = delays_grid - h0

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=4)
    wb = tf.expand_dims(wb, axis=4)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib])

    return out

if __name__=='__main__':

    # test script for the function
    dims = [652,64,140]

    # load the IQ raw data
    I = loadmat('./sample_data/I/sample1.mat')
    I = np.array(I['I'],dtype=np.float32)
    I = np.expand_dims(np.expand_dims(I,0),4)

    Q = loadmat('./sample_data/Q/sample1.mat')
    Q = np.array(Q['Q'], dtype=np.float32)
    Q = np.expand_dims(np.expand_dims(Q, 0), 4)
    img = np.concatenate((I, Q), axis=0)

    input_fmap = tf.placeholder(dtype=tf.float32, shape=[2, 652, 64, 140, 1])
    specs = loadmat('ph_specs.mat')
    theta = np.array(specs['thetaRX']).astype(np.float32)
    elemCoor = loadmat('3Sc_elem_pos.mat')
    elemCoor = tf.constant(np.array(elemCoor['elements_positions'][:, 0]).astype(np.float32))
    specs = specs['specs']

    # get BFed data
    BFfmap = dyanamic_focusing_layer(input_fmap, theta, specs, elemCoor, trainable=True)

    # test gradients with a dummy loss
    loss = tf.reduce_mean(BFfmap - 5.0)
    trainer = tf.train.AdamOptimizer(0.1)
    opt = trainer.minimize(loss)
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    IQ = sess.run(BFfmap, feed_dict={input_fmap:img})
    sess.run(opt, feed_dict={input_fmap: img})
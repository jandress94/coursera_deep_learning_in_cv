
# coding: utf-8

# In[1]:


from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import time
import os
from random import shuffle

from ops import *
from utils import *


# Today we are going to train GAN for generating faces and then we will make fun playing with it. Generative adversarial networks (GANs) are deep neural net architectures comprised of two nets, pitting one against the other (thus the “adversarial”). One neural network, called the generator, generates new faces, while the other, the discriminator,  decides whether each instance of face it reviews belongs to the actual training dataset or not.
# 
# Firstly download aligned faces of celebrities from here <a href="https://yadi.sk/d/xjuClJJH3MAVXh">link</a> and extract them into folder near ipython notebook.

# # Implementation of GAN

# Constant variables below depends on your dataset and choosing of architecture.

# In[2]:


DATA_PATH = './img_align_celeba/' # Path to the dataset with celebA faces
Z_DIM=100 # Dimension of face's manifold
GENERATOR_DENSE_SIZE=64*8 # Length of first tensor in generator

IMAGE_SIZE=64 # Shapes of input image
BATCH_SIZE=64 # Batch size
N_CHANNELS = 3 # Number channels of input image

MERGE_X = 8 # Number images in merged image


# In[3]:


assert(os.path.exists(DATA_PATH)), 'Please, download aligned celebA to DATA_PATH folder'


# Let's define GAN. To do it, we need to define generator, discriminator and loss functions.

# Here are some tips on the architecture of the generator:
# 1. The deeper is convolution, the less filters is using.
# 2. Apply deconvolutions-relu layers to achieve input image shape.
# 3. Use batch normalization before nonlinearity for speed and stability of learning.
# 4. Use tanh activation at the end of network (in this case images should be scaled to [-1, 1])
# 5. To force generator not to collapse and produce different outputs initialize bias with zero (see linear layer).
# 
# Other useful tips: https://github.com/soumith/ganhacks. Example of architecture see below. You may also use defined layers from ops.py. <b> Please, use names for layers started with "g\_" for generator and "d_" for discriminator.</b>

# <img src="https://raw.githubusercontent.com/carpedm20/DCGAN-tensorflow/master/DCGAN.png">

# ## Writing generator function (1 point)

# In[4]:


def generator(z, is_training):
    # Firstly let's reshape input vector into 3-d tensor. 
    z_ = linear(z, GENERATOR_DENSE_SIZE * 4 * 4, 'g_h0_lin')
    h_in = tf.reshape(z_, [-1, 4, 4, GENERATOR_DENSE_SIZE])
    
#     print(h_in)
    
#     h_prev = deconv2d(h_in, [BATCH_SIZE, 4, 4, 1024], name='g_dec_1')
#     h_prev = batch_norm(name="g_bn_1")(h_prev, train=is_training)
#     h_prev = lrelu(h_prev, name="g_lrelu_1")
    
#     print(h_prev)
    
    h_prev = deconv2d(h_in, [BATCH_SIZE, 8, 8, 512], name='g_dec_2')
    h_prev = batch_norm(name="g_bn_2")(h_prev, train=is_training)
    h_prev = lrelu(h_prev, name="g_lrelu_2")
    
    h_prev = deconv2d(h_prev, [BATCH_SIZE, 16, 16, 256], name='g_dec_3')
    h_prev = batch_norm(name="g_bn_3")(h_prev, train=is_training)
    h_prev = lrelu(h_prev, name="g_lrelu_3")
    
    h_prev = deconv2d(h_prev, [BATCH_SIZE, 32, 32, 128], name='g_dec_4')
    h_prev = batch_norm(name="g_bn_4")(h_prev, train=is_training)
    h_prev = lrelu(h_prev, name="g_lrelu_4")
    
    h_out = deconv2d(h_prev, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS],
            name='g_out')

    return tf.nn.tanh(h_out)


# Now let's define discriminator. Discriminator takes 3d tensor as input and outputs one number - probability that this is an image.
# 
# Some advice for discriminator's architecture:
# 1. Use batch normalization between convolutions and nonlinearities.
# 2. Use leaky relu with the leak about 0.2.
# 3. The deeper the layer, the more filters you can use.
# 
# If you use batch normalization, please define every layer in their own scope and pass is_training parameter there. Or you may use class of batch normalization from ops.py. Do not forget to fratten tensor after the convolution blocks.

# ## Writing discriminator function (1 point)

# In[5]:


def discriminator(image, is_training, batch_norms=None):
    h = conv2d(image, 64, name="d_con_0")
    h = batch_norms[0](h, train=is_training)
    h = lrelu(h, name='d_lrelu_0')
    
    h = conv2d(h, 128, name="d_con_1")
    h = batch_norms[1](h, train=is_training)
    h = lrelu(h, name='d_lrelu_1')
    
    h = conv2d(h, 256, name="d_con_2")
    h = batch_norms[2](h, train=is_training)
    h = lrelu(h, name='d_lrelu_2')
    
    h = conv2d(h, 512, name="d_con_3")
    h = batch_norms[3](h, train=is_training)
    h = lrelu(h, name='d_lrelu_3')
    
    h = tf.contrib.layers.flatten(h)
    # print(h)
    
    h = linear(h, 1024, scope="d_lin_0")
    h = lrelu(h, name='d_lrelu_4')
    
    h = linear(h, 64, scope='d_lin_1')
    h = lrelu(h, name='d_lrelu_5')
    
    linear_out = linear(h, 1, scope='d_lin_2')
    
    return tf.nn.sigmoid(linear_out), linear_out


# Now let's define generator and discriminator.

# In[6]:


tf.reset_default_graph()
is_training = tf.placeholder(tf.bool, name='is_training')

with tf.variable_scope("G") as scope:
    z = tf.placeholder(tf.float32, [None, Z_DIM], name='z')
    G = generator(z, is_training)

with tf.variable_scope('D') as scope:
    images = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS])
    
    # If you use batch norms from ops define them here (like batch_norms = [batch_norm(name='d_bn0')])
    # and pass to discriminator function instances.
    batch_norms = [batch_norm(name='d_bn%d' % d) for d in range(4)]
    D_real, D_real_logits = discriminator(images, is_training, batch_norms = batch_norms)
    scope.reuse_variables()
    D_fake, D_fake_logits = discriminator(G, is_training, batch_norms = batch_norms)
    


# 
# Write definition of loss funstions according to formulas:
# $$ D\_loss = \frac{-1}{m} \sum_{i=1}^{m}[\log{D(x_i)} + \log{(1 - D(G(z_i)))}]$$
# $$ G\_loss = \frac{1}{m} \sum_{i=1}^{m} \log{(1 - D(G(z_i)))}$$
# 
# Or for better learning you may try other loss for generator:
# $$ G\_loss = \frac{-1}{m} \sum_{i=1}^{m} \log{(D(G(z_i)))}$$

# ## Writing loss functions (2 points)

# In[19]:


"""
        You code goes here. Define discriminator and generator losses
"""

# from tf.math import log, reduce_sum

d_loss_real = -tf.reduce_mean(tf.log(D_real))

d_loss_fake = -tf.reduce_mean(tf.log(1. - D_fake))

d_loss = d_loss_real + d_loss_fake

g_loss = -tf.reduce_mean(tf.log(D_fake))
# g_loss = tf.reduce_sum(tf.log(1. - D_fake)) / tf.cast(tf.shape(D_fake)[0], np.float32)


# Create optimizers. We use different optimizers for discriminator and generator, so we needed a separate prefix for the discriminator and generator variables (g_ for generator, d_ for disciminator).

# In[20]:


tvars = tf.trainable_variables()
## All variables of discriminator
d_vars = [v for v in tvars if 'd_' in v.name]

## All variables of generator
g_vars = [v for v in tvars if 'g_' in v.name]

LEARNING_RATE = 0.0002 # Learning rate for adam optimizer
BETA = 0.5 # Beta paramater in adam optimizer

##Optimizers - ypu may use your favourite instead.
d_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA).minimize(g_loss, var_list=g_vars) 


# Load data:

# In[21]:


data = glob(os.path.join(DATA_PATH, "*.jpg"))
assert(len(data) > 0), "Length of training data should be more than zero"


# Functions for training and evaluations.

# In[22]:


def load(sess, load_dir):
    """load network's paramaters
    
    load_dir : path to load dir
    """
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(load_dir)
    if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)


# ## Training GAN (1 point + 2 for good results)

# In[25]:


def train(sess, load_dir=None, save_frequency=100, sample_frequency=100, sample_dir='sample_faces',
          save_dir='checkpoint', max_to_keep=1, model_name='dcgan.model',
          n_epochs=25, n_generator_update=2):
    """train gan
    Parameters
    -------------------------------------------
    load_dir : str, default = None
        path to the folder with parameters
    save_frequency: int, default = 100
        how often save parameters []
    sample_frequency: int, default = None (not sample)
        how often sample faces
    sample_dir: str, default = samples
        directory for sampled images
    save_dir: str, default = 'checkpoint'
        path where to save parameters
    max_to_keep: int, default = 1
        how many last checkpoints to store
    model_name: str, default='dcgan.model'
        name of model
    n_epochs: int, default = 25 
        number epochs to train
    n_generator_update: int, default = 2
        how many times run generator updates per one discriminator update
    -------------------------------------------
    """
    
    if save_frequency is not None:
        saver = tf.train.Saver(max_to_keep=max_to_keep)
        
    if load_dir is not None:
        print("Reading checkpoints...")
        load(sess, load_dir)
        print("Loaded checkpoints")
    else:
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

    counter=1
    start_time = time.time()
    
    for epoch in range(n_epochs):
        shuffle(data)

        batch_idxs = min(len(data), np.inf) // BATCH_SIZE
        for idx in range(0, batch_idxs):
            batch_files = data[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
            batch = [get_image(batch_file, IMAGE_SIZE) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            batch_z = np.random.uniform(-1, 1, [BATCH_SIZE, Z_DIM]).astype(np.float32)

            # Update D network
            sess.run(d_optim, feed_dict={images: batch_images, z: batch_z,is_training: True})

            # Update G network
            for _ in range(n_generator_update):
                sess.run(g_optim,
                    feed_dict={z: batch_z, is_training: True})

            counter += 1

            if np.mod(counter, 50) == 1:
                errD_fake = d_loss_fake.eval({z: batch_z, is_training: False})
                errD_real = d_loss_real.eval({images: batch_images, is_training: False})
                errG = g_loss.eval({z: batch_z, is_training: False})

                print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                    epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))

            if np.mod(counter, save_frequency) == 1:
                print("Saved model")
                saver.save(sess, 
                           os.path.join(save_dir, model_name))

            if np.mod(counter, sample_frequency) == 1:
                samples = sess.run(G, feed_dict={z: batch_z, is_training: False} )
                save_images(samples, [MERGE_X, BATCH_SIZE//MERGE_X], os.path.join(sample_dir, 'train_{:02d}_{:04d}.png'.format(epoch, idx)))
                print("Sample")


# In[27]:


with tf.Session() as sess:
    train(sess, save_dir='checkpoint')


# If you generated something that looks like a face - it's cool! Add 2 points to your mark.

# ## Face interpolation (1 point)

# Let's interpolate between faces: generate two vectors $z_1$ and $z_2$ and get a batch of vectors of the form $\alpha\cdot z_1 + (1- \alpha)\cdot  z_2, \alpha \in [0,1].$ Generate faces on them and look at results. The generator displays pictures in the range from -1 to 1, so use the inverse transform function from the file utils.py.

# In[ ]:


# batchz = ...


# ## Adding a smile (1 point + 1 point for good results)

# Let's make face smiling. Find several vectors z, such that the generator generates smiling faces and not. Five vectors in every group should be enough (but the more, the better).

# Calculate "smile vector" as mean of vectors z with generated smile on it minus mean of vectors z with generated not smile on it.

# Look at the result of applying the smile vector: compare the results of generation before and after the addition of the smile vector.

# If faces looks really cool, add bonus 1 point to your score.

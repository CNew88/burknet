# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 19:02:40 2018

@author: Burkay
"""

import gzip
import _pickle as pickle
import matplotlib.pyplot as pp
import numpy as np
import PIL as pl
import tensorflow as tf
import datetime
import os

def convert_from_float_to_uint8(dat):
  if dat.dtype != 'uint8':
    dat = dat*255
    dat[dat > 255] = 255
    dat[dat < 0] = 0
    dat = dat.astype('uint8')
  return dat

def create_canvas(dat, ncolumns=-1, nrows=-1):
  # Support for 3 dimensions 
  if len(dat.shape) == 3:
    Ns = 1
    Ny = dat.shape[0]
    Nx = dat.shape[1]
    Nc = dat.shape[2]
    dat = np.reshape(dat, [1, Ny, Nx, Nc])
  else:
    Ns = dat.shape[0]
    Ny = dat.shape[1]
    Nx = dat.shape[2]
    Nc = dat.shape[3]

  # Support for float data
  dat = convert_from_float_to_uint8(dat)

  # Special treat single image
  if Ns == 1:
    nrows = 1
    ncolumns = 1
  else:
    # Auto rows/columns
    if nrows == -1 and ncolumns == -1:
      ncolumns = int(np.ceil(np.sqrt((Ns*5)/3)))
      nrows = 1+(Ns-1)//ncolumns
    elif nrows == -1:
      nrows = 1+(Ns-1)//ncolumns
    elif ncolumns == -1:
      ncolumns = 1+(Ns-1)//nrows
    
  # Construct canvas
  canvas = np.zeros((nrows*Ny, ncolumns*Nx, Nc), dtype='uint8')
  for i in range(Ns):
    ic = i % ncolumns
    ir = i // ncolumns
    canvas[ir*Ny:(ir+1)*Ny,ic*Nx:(ic+1)*Nx,:] = dat[i, :, :, :]
  if Nc == 1:
    canvas = np.squeeze(canvas, axis=2)
  
  # Return canvas
  return canvas

def load_image(filename):
  img = pl.Image.open(filename)
  pix = np.array(list(img.getdata()))
  arr = np.reshape(pix,[img.height, img.width, pix.shape[1]]).astype('float')
  return arr

# Plot data
# dat: BHWC, C=1, 3 or 4
def plot(dat, ncolumns=-1, nrows=-1):
  # Create canvas
  canvas = create_canvas(dat, ncolumns, nrows)
  
  # Show canvas
  pp.imshow(canvas)
  ax = pp.gca()
  for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(5)

# Save data as canvas
# dat: BHWC, C=1, 3 or 4
# filename: name of file including desired extension
def save_as_canvas(dat, filename, ncolumns=-1, nrows=-1):
  # Create canvas
  canvas = create_canvas(dat, ncolumns, nrows)
  
  # Save as canvas
  im = pl.Image.fromarray(canvas)
  im.save(filename)

# Save nn-forward-flow-data as canvas
# dat: list of 1HWC, each item may have different H, W, or C
# filename: name of file including desired extension
def save_as_flow(datlist, filename, whiten=1):
  # Add all heights and max all widths x channels
  h = 0
  w = 0
  hseparator = 1
  for d in datlist:
    h += d.shape[1] + hseparator
    w = np.max((d.shape[2]*d.shape[3], w))
  
  def stack(X):
    N = X.shape[2]
    Xs = np.zeros((X.shape[0], X.shape[1]*N), dtype=X.dtype)
    for i in range(N):
      Xs[:, i*X.shape[1]:(i+1)*X.shape[1]] = X[:, :, i]
    return Xs
  
  # Create canvas and paste items on canvas
  canvas = np.ones((h, w), dtype='uint8')
  h = 0
  for d in datlist:
    # Support for float data
    dat = d[0, :, : ,:]
    if whiten:
      wdat = 0.5 + (dat - np.mean(dat))/(np.std(dat)+1e-10)  #whiten for plot
    else:
      wdat = dat
    sdat = stack(convert_from_float_to_uint8(wdat))
    canvas[h:h+d.shape[1], 0:d.shape[2]*d.shape[3]] = sdat
    canvas[h+d.shape[1]:h+d.shape[1]+ hseparator, :] = 128
    h += d.shape[1] + hseparator
    
  # Save as canvas
  im = pl.Image.fromarray(canvas)
  im.save(filename)

# Batch object
class Batch:
  # Construct data (X) and labels (y)
  def __init__(self, i, Xbatch, ybatch, isample):
    self.i = i
    self.Xbatch = Xbatch
    self.ybatch = ybatch
    self.isample = isample

# Data object
class Data:
  def __init__(self):
    pass
  
  # Construct data (X) and labels (y) and image size Nx x Ny
  def create(self, Xtrain, ytrain, Xtest, ytest, Xdev, ydev,
               m=None, s=None):
    self.Xtrain = Xtrain
    self.ytrain = ytrain
    self.Xtest = Xtest
    self.ytest = ytest
    self.Xdev = Xdev
    self.ydev = ydev
    self.m = m
    self.s = s
    self.ifeedorder = np.arange(self.Xtrain.shape[0]) # Fed to batch generator
    self.active = np.ones(self.Xtrain.shape[0], dtype='bool') # sample active flags

  # Get a batch generator
  def get_batch_generator(self, n_batch):
    iactive = self.ifeedorder[self.active[self.ifeedorder]]
    n_batches = iactive.size // n_batch
    
    for i in range(n_batches):
      yield Batch(i, self.Xtrain[iactive[i*n_batch:(i+1)*n_batch], ...],
                     self.ytrain[iactive[i*n_batch:(i+1)*n_batch], ...],
                     iactive[i*n_batch:(i+1)*n_batch])

  def load(self, filename):
    with gzip.open(filename, 'rb') as f:
      obj = pickle.load(f, encoding='latin1')
    self.create(obj.Xtrain, obj.ytrain, obj.Xtest, obj.ytest,
                obj.Xdev, obj.ydev, obj.m, obj.s)
    
  def save(self, filename):
    with gzip.open(filename,'wb') as f:
      pickle.dump(self,f, protocol=4)

  def shuffle(self):
    np.random.shuffle(self.ifeedorder)
  
# Batch object
class VAE:
  def __init__(self):
    pass

  def create_FC(self, data, n_hidden=[500, 500], n_latent=20):

    # Clear dead nodes, so that no problems with save later
    tf.reset_default_graph()

    ny = data.Xtrain.shape[1]
    nx = data.Xtrain.shape[2]
    nc = data.Xtrain.shape[3]
    self.X=tf.placeholder(dtype=tf.float32, shape=(None, ny, nx, nc))
    self.Xnorm = self.X / 255
    weights_initializer=tf.contrib.layers.variance_scaling_initializer()

    self.Xflat = tf.reshape(self.Xnorm, shape=(tf.shape(self.Xnorm)[0], ny*nx*nc))

    self.flayers = [self.Xflat]

    for i in range(len(n_hidden)):
      self.flayers.append(tf.contrib.layers.fully_connected(self.flayers[-1], n_hidden[i],
                          activation_fn=tf.nn.elu, weights_initializer=weights_initializer))
    
    self.latent_mean = tf.contrib.layers.fully_connected(self.flayers[-1], n_latent,
                  activation_fn=None, weights_initializer=weights_initializer)
    self.latent_gamma = tf.contrib.layers.fully_connected(self.flayers[-1], n_latent,
                  activation_fn=None, weights_initializer=weights_initializer)
    self.latent_sigma = tf.exp(0.5 * self.latent_gamma)
    self.noise = tf.random_normal(tf.shape(self.latent_sigma), dtype=tf.float32)
    self.latent = self.latent_mean + self.latent_sigma * self.noise

    self.rlayers = [self.latent]

    rn_hidden = [ny*nx*nc] + n_hidden
    for i in range(len(rn_hidden)):
      if i != len(rn_hidden)-1:
        activation_fn = tf.nn.elu
      else:
        activation_fn = None
      self.rlayers.append(tf.contrib.layers.fully_connected(self.rlayers[-1], rn_hidden[-1-i],
                    activation_fn=activation_fn, weights_initializer=weights_initializer))
    self.logits = tf.reshape(self.rlayers[-1], shape=(tf.shape(self.rlayers[-1])[0], ny, nx, nc))
    self.outputs = tf.sigmoid(self.logits)

  def create_C(self, data, n_latent=20, beta=1):

    # Clear dead nodes, so that no problems with save later
    tf.reset_default_graph()

    ny = data.Xtrain.shape[1]
    nx = data.Xtrain.shape[2]
    nc = data.Xtrain.shape[3]
    # na = data.ytrain.shape[1]

    with tf.device('/device:GPU:0'):
      self.X=tf.placeholder(dtype=tf.float32, shape=(None, ny, nx, nc))
      self.Xnorm = self.X / 255
      weights_initializer=tf.contrib.layers.variance_scaling_initializer()
      weights_initializer_relu=tf.random_normal_initializer(mean=0, stddev=1e-2)
      lrelu = tf.nn.leaky_relu
      
      maps = [32, 64, 128, 256]
      filt = [4,  4,   4,  4]
      pool = [2,  2,   2,  2]
  
      self.flayers = [self.Xnorm]
  
      self.training = tf.placeholder(dtype=tf.bool, shape=())
      for i in range(len(maps)):
        self.flayers.append(tf.contrib.layers.conv2d(self.flayers[-1], num_outputs=maps[i], kernel_size=filt[i], padding="same",
                                                     stride=pool[i], activation_fn=None, weights_initializer=weights_initializer_relu))
        #self.flayers.append(tf.contrib.layers.max_pool2d(self.flayers[-1], kernel_size=pool[i], stride=pool[i]))
        #self.flayers.append(tf.contrib.layers.batch_norm(self.flayers[-1], is_training=self.training, updates_collections=None))
        self.flayers.append(lrelu(self.flayers[-1]))
        
      Nscale = np.prod(np.array(pool))
      Nflattened = int(maps[-1]*nx*ny//(Nscale**2))
      self.flattened = tf.reshape(self.flayers[-1], shape=(tf.shape(self.flayers[-1])[0], Nflattened))
  
      '''
      self.labels = tf.placeholder(dtype=tf.float32, shape=(None, na))
      self.attributes = tf.contrib.layers.fully_connected(self.flattened, na,
                        activation_fn=tf.sigmoid, weights_initializer=weights_initializer)
      '''
  
  
      self.latent_mean  = tf.contrib.layers.fully_connected(self.flattened, n_latent,
                    activation_fn=None, weights_initializer=weights_initializer)
      self.latent_gamma = tf.contrib.layers.fully_connected(self.flattened, n_latent,
                    activation_fn=None, weights_initializer=weights_initializer)
      self.latent_sigma = tf.exp(0.5 * self.latent_gamma)
      self.noise = tf.random_normal(tf.shape(self.latent_sigma), dtype=tf.float32)
      self.latent = self.latent_mean + self.latent_sigma * self.noise

    with tf.device('/device:GPU:0'):

      '''
      self.merged = tf.concat((self.latent, self.attributes), axis=1)
      '''
  
      self.unlatent = tf.contrib.layers.fully_connected(self.latent, Nflattened,
                          activation_fn=None, weights_initializer=weights_initializer)
      self.unflattened = tf.reshape(self.unlatent, shape=(tf.shape(self.unlatent)[0], ny//Nscale, nx//Nscale, maps[-1]))
      self.rlayers = [self.unflattened]
  
      rmaps = [nc] + maps
      for i in range(len(maps)):
        self.rlayers.append(tf.image.resize_nearest_neighbor(self.rlayers[-1],
                        size=(self.rlayers[-1].shape[1].value*pool[-i-1],
                              self.rlayers[-1].shape[2].value*pool[-i-1]), align_corners=True))
        self.rlayers.append(tf.contrib.layers.conv2d_transpose(self.rlayers[-1],
                            num_outputs=rmaps[-i-2], kernel_size=filt[-i-1], padding="same",
                            activation_fn=None, weights_initializer=weights_initializer_relu))
  
        self.rlayers.append(lrelu(self.rlayers[-1]))
        self.rlayers.append(tf.contrib.layers.conv2d_transpose(self.rlayers[-1],
                            num_outputs=rmaps[-i-2], kernel_size=filt[-i-1], padding="same",
                            activation_fn=None, weights_initializer=weights_initializer_relu))
  
        if i != len(maps)-1:
          #self.rlayers.append(tf.contrib.layers.batch_norm(self.rlayers[-1], is_training=self.training, updates_collections=None))
          self.rlayers.append(lrelu(self.rlayers[-1]))
        
      self.logits = self.rlayers[-1]
      #self.outputs = tf.sigmoid(self.logits)
      self.outputs = self.logits
      
      # Reconstruction loss
      self.reconstruction_loss_batch = tf.reduce_mean(tf.abs(self.Xnorm - self.outputs), axis=(1, 2, 3))\
              + tf.reduce_mean(tf.abs(self.flayers[2] - self.rlayers[-5]), axis=(1, 2, 3))\
              + tf.reduce_mean(tf.abs(self.flayers[4] - self.rlayers[-10]), axis=(1, 2, 3))\
              + tf.reduce_mean(tf.abs(self.flayers[6] - self.rlayers[-15]), axis=(1, 2, 3))\
              + tf.reduce_mean(tf.abs(self.flayers[8] - self.rlayers[-20]), axis=(1, 2, 3))
      self.reconstruction_loss = tf.reduce_mean(self.reconstruction_loss_batch)

      # Edge loss
      self.XedgeH = self.Xnorm - tf.concat((self.Xnorm[:, -1:, :, :], self.Xnorm[:, 0:-1, :, :]), axis=1)
      self.XedgeW = self.Xnorm - tf.concat((self.Xnorm[:, :, -1:, :], self.Xnorm[:, :, 0:-1, :]), axis=2)
      self.oedgeH = self.outputs - tf.concat((self.outputs[:, -1:, :, :], self.outputs[:, 0:-1, :, :]), axis=1)
      self.oedgeW = self.outputs - tf.concat((self.outputs[:, :, -1:, :], self.outputs[:, :, 0:-1, :]), axis=2)
      self.edge_loss_batch = tf.reduce_mean(tf.abs(self.XedgeH - self.oedgeH) +
                                      tf.abs(self.XedgeW - self.oedgeW), axis=(1, 2, 3))
      self.edge_loss = tf.reduce_mean(self.edge_loss_batch)

      '''
      # Reconstruction loss
      self.reconstruction_loss = tf.reduce_mean(tf.squared_difference(self.Xnorm, self.outputs))\
              + tf.reduce_mean(tf.squared_difference(self.flayers[2], self.rlayers[-5]))\
              + tf.reduce_mean(tf.squared_difference(self.flayers[4], self.rlayers[-10]))\
              + tf.reduce_mean(tf.squared_difference(self.flayers[6], self.rlayers[-15]))\
              + tf.reduce_mean(tf.squared_difference(self.flayers[8], self.rlayers[-20]))

      # Edge loss
      self.XedgeH = self.Xnorm - tf.concat((self.Xnorm[:, -1:, :, :], self.Xnorm[:, 0:-1, :, :]), axis=1)
      self.XedgeW = self.Xnorm - tf.concat((self.Xnorm[:, :, -1:, :], self.Xnorm[:, :, 0:-1, :]), axis=2)
      self.oedgeH = self.outputs - tf.concat((self.outputs[:, -1:, :, :], self.outputs[:, 0:-1, :, :]), axis=1)
      self.oedgeW = self.outputs - tf.concat((self.outputs[:, :, -1:, :], self.outputs[:, :, 0:-1, :]), axis=2)
      self.edge_loss = tf.reduce_mean(tf.squared_difference(self.XedgeH, self.oedgeH) +
                                      tf.squared_difference(self.XedgeW, self.oedgeW))
      '''
  
      # Latent loss
      self.eta = tf.Variable(0.3, dtype=tf.float32, trainable=False)
      self.latent_loss_batch = self.eta * beta * tf.reduce_mean(tf.exp(self.latent_gamma)
                                   + tf.square(self.latent_mean) - 1 - self.latent_gamma, axis=1) / 46
      self.latent_loss = tf.reduce_mean(self.latent_loss_batch)
  
      '''
      # Attribute loss
      self.attribute_loss = alpha * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.attributes))
      self.attribute_accuracy = tf.count_nonzero(
               tf.abs(self.labels-self.attributes) > 0.5) / tf.size(self.labels, out_type=tf.int64)
      self.cost = self.reconstruction_loss + self.latent_loss + self.attribute_loss
      '''
          
      self.cost = self.reconstruction_loss + self.edge_loss + self.latent_loss

  def fit(self, data, learning_rate=0.0005, batch_size=64, n_epochs=50,
          sessionname=None, sessionfile=None):
    # Handle session file name
    if sessionname is None:
      # Session filename based on date & time
      sessionname = datetime.datetime.now().strftime('S_%y%m%d_%H%M')
      
    # Set session file name
    if sessionfile is None:
      sessionfile = os.path.join(sessionname, sessionname)

    # Create directory for snapshots
    if not os.path.exists(sessionname):
      os.makedirs(sessionname)
      epochstart = 0
    else:
      epochstart = len([fn for fn in os.listdir(sessionname) if fn.startswith('random')]) - 1
          
    # Optimizer
    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    self.training_op = self.optimizer.minimize(self.cost)

    # Initializer
    self.initializer = tf.global_variables_initializer()
    
    # Saver
    self.saver = tf.train.Saver()
  
    # Train
    with tf.Session() as sess:
      self.initializer.run()
      if epochstart != 0:
        self.saver.restore(sess, sessionfile)
      for epoch in range(epochstart, n_epochs):
        
        # Save
        self.saver.save(sess, sessionfile)
        
        # Generate level snapshot
        outputs_val = self.generateLevels(n_levels=10, session=sess)
        filename = os.path.join(sessionname, 'level_' + "%03d" % epoch + '.png')
        save_as_canvas(outputs_val, filename, ncolumns=self.latent.shape[1].value)
        
        # Generate random snapshot
        ncolumns = 20
        nrows = 12
        outputs_val = self.generateRandom(n_images=nrows*ncolumns, session=sess)
        filename = os.path.join(sessionname, 'random_' + "%03d" % epoch + '.png')
        save_as_canvas(outputs_val, filename, ncolumns=ncolumns, nrows=nrows)
        
        # Visualize data-flow snapshot
        i = 1
        filename = os.path.join(sessionname, 'exampleflow_white_' + "%03d" % epoch + '.png')
        self.saveDataFlow(data.Xtrain[i:i+1,:,:,:], filename, whiten=True, session=sess)
        filename = os.path.join(sessionname, 'exampleflow_' + "%03d" % epoch + '.png')
        self.saveDataFlow(data.Xtrain[i:i+1,:,:,:], filename, whiten=False, session=sess)

        # Visualize input data snapshot
        if epoch == 0:
          ncolumns = 20
          nrows = 12
          i = np.arange(nrows*ncolumns)
          outputs_val_input = data.Xtrain[i, :, :, :]
          filename = os.path.join(sessionname, 'dataref.png')
          save_as_canvas(outputs_val_input, filename, ncolumns=ncolumns, nrows=nrows)

        # Visualize reconstructed data snapshot
        ncolumns = 20
        nrows = 12
        i = np.arange(nrows*ncolumns)
        inputs = data.Xtrain[i, :, :, :]
        outputs_val_rec = self.reconstructData(inputs, session=sess)
        filename = os.path.join(sessionname, 'datareconst_' + "%03d" % epoch + '.png')
        save_as_canvas(outputs_val_rec, filename, ncolumns=ncolumns, nrows=nrows)

        # Visualize edge intensity
        def edge_filter(dat):
          return np.sqrt((dat - np.roll(dat, shift=(0, -1, 0, 0),
                                        axis=(0, 1, 2, 3))) ** 2 +
                         (dat - np.roll(dat, shift=(0,  0,-1, 0),
                                        axis=(0, 1, 2, 3))) ** 2)
        outputs_val_input = data.Xtrain[i, :, :, :]
        edges_val = np.concatenate((edge_filter(outputs_val_input.astype('float')/255),
                                    edge_filter(outputs_val_rec)), axis=2)
        filename = os.path.join(sessionname, 'edgemag_' + "%03d" % epoch + '.png')
        save_as_canvas(edges_val, filename, ncolumns=ncolumns, nrows=nrows)

        # Apply warm-up schedule
        epochstart_wu = 0
        epochtransition = 20
        betaboost = 10
        eta = 1 + (betaboost - 1) * np.max((0, np.min((1,
                               1 - (epoch - epochstart_wu)/epochtransition))))
        #lperiod = 5
        #eta = 1.0 - 0.7*((epoch//lperiod) % 2)

        # Training update
        loss_rec = np.zeros(data.Xtrain.shape[0])
        loss_edg = np.zeros(data.Xtrain.shape[0])
        loss_lat = np.zeros(data.Xtrain.shape[0])
        isample = np.zeros(data.Xtrain.shape[0], dtype='int')
        N = 0
        for batch in data.get_batch_generator(batch_size):
          [_, rval, dval, lval] = sess.run([self.training_op,
                self.reconstruction_loss_batch, self.edge_loss_batch, self.latent_loss_batch],
                   feed_dict={self.X: batch.Xbatch, self.training: True,
                              self.eta: eta})
          Nb = batch.Xbatch.shape[0]
          ib = np.arange(N, N+Nb)
          loss_rec[ib] = rval
          loss_edg[ib] = dval
          loss_lat[ib] = lval
          isample[ib] = batch.isample
          N += Nb
        
        # Print stats
        print("epoch " + "%3d" % epoch + '/' + "%3d" % n_epochs +
              ", reconst_loss = " + "%1.6f" % np.mean(loss_rec[0:N]) +
              ", edge_loss = " + "%1.6f" % np.mean(loss_edg[0:N]) +
              ", latent_loss = " + "%1.6f" % np.mean(loss_lat[0:N]))

        # NaN recovery
        if np.any(np.isnan(loss_rec)):
          print('Encountered NaN, restoring')
          self.saver.restore(sess, sessionfile)

        # Deactivate data points
        epoch_dw = 25
        percentage_to_ignore = 50
        if epoch == epoch_dw:
          N_ignore = (N * percentage_to_ignore) // 100
          loss = loss_rec + loss_edg + loss_lat
          iorder = np.argsort(loss)
          data.active[:] = True
          data.active[isample[iorder[-N_ignore:]]] = False

  def generateLevels(self, n_levels, session='./tempsession'):
    # Generation params
    n_latent = self.latent.shape[1].value
    n_images = n_latent * n_levels
    test_codes = np.zeros((n_images, n_latent))
    for i in range(n_images):
      i_level = i // n_latent
      i_code = i % n_latent
      test_codes[i, i_code] = 10*(i_level-n_levels/2+0.5)/n_levels
    codings_rnd = test_codes
    #codings_rnd = np.random.normal(size=[n_images, n_latent])
        
    # Generate
    if isinstance(session, str):
      with tf.Session() as sess:
        self.saver.restore(sess, session)
        outputs_val = self.outputs.eval(feed_dict={self.latent: codings_rnd, self.training: False})
    else:
      outputs_val = self.outputs.eval(session=session, feed_dict={self.latent: codings_rnd, self.training: False})
    return outputs_val

  def generateRandom(self, n_images, session='./tempsession'):
    # Generation params
    n_latent = self.latent.shape[1].value
    codings_rnd = np.random.normal(size=[n_images, n_latent])
        
    # Generate
    if isinstance(session, str):
      with tf.Session() as sess:
        self.saver.restore(sess, session)
        outputs_val = self.outputs.eval(feed_dict={self.latent: codings_rnd, self.training: False})
    else:
        outputs_val = self.outputs.eval(session=session, feed_dict={self.latent: codings_rnd, self.training: False})
    return outputs_val

  def saveDataFlow(self, data_pt, filename, whiten, session='./tempsession'):
    if isinstance(session, str):
      with tf.Session() as sess:
        self.saver.restore(sess, session)
        imgs = sess.run(self.flayers + self.rlayers + [self.outputs], feed_dict={self.X: data_pt, self.training: False})
    else:
      imgs = session.run(self.flayers + self.rlayers + [self.outputs], feed_dict={self.X: data_pt, self.training: False})
      
    # Save data flow
    save_as_flow(imgs, filename, whiten)
      
  def reconstructData(self, data, session='./tempsession'):
    if isinstance(session, str):
      with tf.Session() as sess:
        self.saver.restore(sess, session)
        imgs = self.outputs.eval(feed_dict={self.X: data, self.training: False})
    else:
      imgs = self.outputs.eval(session=session, feed_dict={self.X: data, self.training: False})
    return imgs
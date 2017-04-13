# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import tensorflow.python.platform

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from scipy.ndimage import zoom

from util.util_funcs import nhot, onehot


def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels


class Dataloader(object):

  def __init__(self, data, sizes,norm):
    """Construct a DataSet.
    """
    self.data = {}
    self.sizes = sizes
    self._index_in_epoch = 0
    self._epochs_completed = 0

    self.logging_pol=None

    ds = ['train','val','test']
    self.mean = None
    self.std = None
    for d in ds:
        X,y  = data[d]

        X = X.copy().astype(np.float32)
        y = y.copy().astype(np.int16)
        if norm:
            if not self.mean:
                self.mean = np.mean(X)
                self.std = np.sqrt(np.var(X)+1E-09)
            X -= self.mean
            X /= self.std
        self.data[d] = [X,y]
    return

  @property
  def im_size(self):
      _,H,W,_ = self.data['train'][0].shape
      return H, W

  def unnorm(self,X):
      return (X*self.std)+self.mean

  @property
  def pol0(self):
      return self.logging_pol

  @pol0.setter
  def pol0(self,pol):
      self.logging_pol = pol

  def next_batch(self, batch_size = 32,dataset='train'):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self.sizes[dataset]:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self.sizes[dataset])
      numpy.random.shuffle(perm)
      self.data[dataset][0] = self.data[dataset][0][perm]
      self.data[dataset][1] = self.data[dataset][1][perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self.sizes[dataset]
    end = self._index_in_epoch
    return self.data[dataset][0][start:end], self.data[dataset][1][start:end]

  def next_lb_batch(self,batch_size,dataset='train'):
      #wrapper of next_batch() to generate logged bandit feedback
      if not self.pol0:
          logging_pol = np.array([0.1]*10)
      else:
          logging_pol = self.pol0
      X,correct = self.next_batch(batch_size,dataset)
      action = np.random.choice(10,size=(batch_size,),replace=True,p=logging_pol)
      reward = np.equal(action,correct).astype(np.float32)
      prob = np.array([logging_pol[a] for a in action]).astype(np.float32)
      return X,action,reward-0.1,prob


def read_data_sets(train_dir, one_hot=False,norm=False):
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 5000

  #TRAIN
  local_file = maybe_download(TRAIN_IMAGES, train_dir)
  train_images = extract_images(local_file)
  local_file = maybe_download(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)

  #TEST
  local_file = maybe_download(TEST_IMAGES, train_dir)
  test_images = extract_images(local_file)
  local_file = maybe_download(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)

  #VALIDATION
  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]


  data = {  'train'     :(train_images, train_labels),
            'val'       :(validation_images, validation_labels),
            'test'      :(test_images, test_labels)}
  sizes = {'train'      :train_images.shape[0],
            'val'       :validation_images.shape[0],
            'test'      :test_images.shape[0]}
  return Dataloader(data, sizes, norm)

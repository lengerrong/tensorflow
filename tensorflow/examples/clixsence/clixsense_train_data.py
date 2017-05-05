# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A binary to train ClixSense using a single GPU.

Accuracy:
clixsense_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by clixsense_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
from threading import Thread
from PIL import Image


import clixsense
import numpy as np
import os
import time
import tensorflow as tf
import subprocess

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/clixsense_train_data',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    images, labels = clixsense.distorted_inputs()
   
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = clixsense.inference(images)

    # Calculate loss.
    loss = clixsense.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = clixsense.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)

def image_gen(filepath, label):
  try:
    im = Image.open(filepath)
    im = (np.array(im))

    r = im[:,:,0].flatten()
    g = im[:,:,1].flatten()
    b = im[:,:,2].flatten()

    labels = [label]
    out  = np.array(list(labels) + list(r) + list(g) + list(b), np.uint8)
    binpath = filepath + '.bin'
    binf = open(binpath, "w")
    out.tofile(binf)
    binf.close()
    return binpath
  except Exception, e:
    print (e)
    return ''
  
def main(argv):  # pylint: disable=unused-argument
  folder = argv[0]
  if (len(argv) > 1):
    if tf.gfile.Exists(argv[1]):
      folder = argv[1]
    else:
      print ('%s not existed' % argv[1])
      return
  else:
    print ("please input training image data folder path")
    return
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  for f in tf.gfile.ListDirectory(folder):
    filepath = os.path.join(folder, f)
    try:
      im = Image.open(filepath)
      p = subprocess.Popen(["display", filepath])
      label = raw_input("please label the image, 0 means cat, 1 means dog:")
      p.kill()
      binpath = image_gen(filepath, label)
      with tf.Graph().as_default():
        images, labels = clixsense.distorted_inputs_one(binpath)
        global_step = tf.contrib.framework.get_or_create_global_step()
        logits = clixsense.inference_one(images)
        loss = clixsense.loss(logits, labels)
        #top_k_op = tf.nn.in_top_k(logits, labels, 1)
        #variable_averages = tf.train.ExponentialMovingAverage(
        #clixsense.MOVING_AVERAGE_DECAY)
        #variables_to_restore = variable_averages.variables_to_restore()
        #saver = tf.train.Saver(variables_to_restore)
        #with tf.Session() as sess:
        #  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        #  if ckpt and ckpt.model_checkpoint_path:
        #    # Restores from checkpoint
        #    saver.restore(sess, ckpt.model_checkpoint_path)
        #    predictions = sess.run([top_k_op])
        #    print (predictions)

        train_op = clixsense.train(loss, global_step)
        class _LoggerHook(tf.train.SessionRunHook):
          """Logs loss and runtime."""

          def begin(self):
            self._step = -1
            self._start_time = time.time()

          def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs(loss)  # Asks for loss value.

          def after_run(self, run_context, run_values):
            if self._step % FLAGS.log_frequency == 0:
              current_time = time.time()
              duration = current_time - self._start_time
              self._start_time = current_time

              loss_value = run_values.results
              examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
              sec_per_batch = float(duration / FLAGS.log_frequency)

              format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
              print (format_str % (datetime.now(), self._step, loss_value,
                                   examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()],
            config=tf.ConfigProto(
                log_device_placement=FLAGS.log_device_placement)) as mon_sess:
          while not mon_sess.should_stop():
            mon_sess.run(train_op)
    except Exception, e:
      print (e)

if __name__ == '__main__':
  tf.app.run()

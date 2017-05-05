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

"""Evaluation for ClixSense

Accuracy:
clixsense_train.py achieves 25.0% accuracy after 100K steps (256 epochs
of data) as judged by clixsense_eval.py.

Speed:
On a single Tesla K40, clixsense_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~25%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
import os
import subprocess
import clixsense

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/clixsense_train',
                           """Directory where to read model checkpoints.""")

def eval_once(saver, logits):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/clixsense_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
 
    logits_precesion = ''

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      num_iter = 1
      step = 0

      while step < num_iter and not coord.should_stop():
        logits_precesion = (sess.run(logits))
        step += 1

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    return logits_precesion
        
def evaluate(filepath):
  """Eval ClixSense for a number of steps."""
  data = clixsense.image_gen(filepath, 0)
  if (data.status == clixsense.IMAGE_GEN_FAIL):
    return
  datapath = filepath + '.bin'
  data.imagebytes.tofile(datapath)
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    images, labels = clixsense.distorted_inputs_one(datapath)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = clixsense.inference_one(images)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        clixsense.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    logits_precesion = eval_once(saver, logits)
    p = subprocess.Popen(["display", filepath])
    raw_input("precesion result:\n\t%f cat\n\t%f dog\npress enter key to quit......" % (logits_precesion[0][0], logits_precesion[0][1]))
    p.kill()
    
  os.remove(datapath)

def main(argv):  # pylint: disable=unused-argument
  clixsense.maybe_download_and_extract()
  if (len(argv) >= 2 and os.path.exists(argv[1])):
    evaluate(argv[1])
  else:
    print ("Usage : ")
    print ('python %s image_file' % argv[0])

if __name__ == '__main__':
  tf.app.run()

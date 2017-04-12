# Similar format to cifar-10 dataset
# https://www.cs.toronto.edu/~kriz/cifar.html

# The first byte is the label of the first image,
# which is a number in the range 0-1, label 0 means cat, 
# label 1 means dog, no other labels.
# 
# The image size is 128 x 96. The next 36864 bytes are the
# values of the pixels of the image. The first 12288 bytes are the
# red channel values, the next 12288 the green, and the final 12288
# the blue. The values are stored in row-major order, so the first
# 128 bytes are the red channel values of the first row of the image.

from PIL import Image
import numpy as np
import os
import sys

def append_clixsense_dataset(filename, dataset, iscat):
  """Reads image and convert to cifar-10 dataset format and append to dataset

  Args:
    filename: a image path with the filename to read from
    dataset: an open file object to save to

  Returns:
    None
  """

  im = Image.open(filename)
  im = (np.array(im))

  r = im[:,:,0].flatten()
  g = im[:,:,1].flatten()
  b = im[:,:,2].flatten()

  if iscat:
    label = [0]
  else:
    label = [1]

  out  = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
  out.tofile(dataset)

NUM_MAX_EXAMPLES_PER_DATA_BATCH = 3000

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print "Usage : "
    print "python clixsense_dataset_gen.py data_dir"
    sys.exit(1)

  # args 1 is the folder path where clixsense images located.
  data_dir = sys.argv[1]
  if not os.path.exists(data_dir):
    print data_dir + ' not existed '
    sys.exit(2);

  eval_data = 0
  if len(sys.argv) > 2:
    eval_data = sys.argv[2]

  # dataset name is clixsense_data_batch.bin in the same folder
  if not eval_data:
    dataset = data_dir + '/data_batch_1.bin'
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    dataset = data_dir + '/test_batch.bin'

  for f in filenames:
    if os.path.exists(f):
      os.remove(f)

  dataset_f = open(dataset, "w+")
  if not dataset_f:
    print "unable to open " + dataset + " with w+ mode"
    sys.exit(3)

  cc = 0
  dc = 1
  for lists in os.listdir(data_dir):
    path = os.path.join(data_dir, lists)
    if lists.find("cat") >= 0:
      iscat = 1
    else:
      iscat = 0
    if os.path.splitext(lists)[1] == ".jpg":
      if not eval_data:
        if (dc < 5 and cc >= NUM_MAX_EXAMPLES_PER_DATA_BATCH):
          cc = 0
          dc = dc + 1
          dataset_f.close()
          dataset = data_dir + '/data_batch_' + str(dc) + '.bin'
          dataset_f = open(dataset, "w+")
        cc = cc + 1
      append_clixsense_dataset(path, dataset_f, iscat)
  dataset_f.close()

  sys.exit(0)

from PIL import Image
import numpy as np
import os
import subprocess
import sys

IMAGE_GEN_OK = True
IMAGE_GEN_FAIL = False

def image_gen(filepath, label):
  class ImageGen(object):
    pass
  result = ImageGen()
  result.status = IMAGE_GEN_FAIL
  try:
    im = Image.open(filepath)
    im = (np.array(im))

    r = im[:,:,0].flatten()
    g = im[:,:,1].flatten()
    b = im[:,:,2].flatten()

    labels = [label]
    result.imagebytes = np.array(list(labels) + list(r) + list(g) + list(b), np.uint8)
    result.status = IMAGE_GEN_OK
  except Exception, e:
    print (e)
  return result

def main(argv):
  folder = argv[0]
  if (len(argv) > 1):
    if os.path.exists(argv[1]):
      folder = argv[1]
    else:
      print ('%s not existed' % argv[1])
      return
  else:
    print ("Usage : ")
    print ("python %s data_dir" % argv[0])
    return

  NUM_MAX_EXAMPLES_PER_DATA_BATCH = 3000
  dataset = os.path.join(folder, 'data_batch_1.bin')
  filenames = [os.path.join(folder, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]

  dataset_f = open(dataset, "a+")
  if not dataset_f:
    print "unable to open " + dataset + " with w+ mode"
    return
  cc = 0
  dc = 1
  ii = 0
  cat = 0
  dog = 0
  nor = 0
  for f in os.listdir(folder):
    filepath = os.path.join(folder, f)
    try:
      imf = open(filepath, "r")
      im = Image.open(imf)
      imf.close()
      p = subprocess.Popen(["display", filepath])
      ii = ii + 1
      label = raw_input("please label the %d image:\n\t0 means cat:\n\t1 means dog:\n\t-1 not a cat or dog:\n" % ii)
      p.kill()
      if (label == '-1'):
        os.remove(filepath)
        nor = nor + 1
        continue
      elif (label == '1'):
        dog = dog + 1
      elif (label == '0'):
        cat = cat + 1
      print ("generate data for %s" % filepath)
      result = image_gen(filepath, label)
      os.remove(filepath)
      if (result.status == IMAGE_GEN_OK):
        if (dc < 5 and cc >= NUM_MAX_EXAMPLES_PER_DATA_BATCH):
          cc = 0
          dc = dc + 1
          dataset_f.close()
          dataset_f.close()
          dataset = folder + '/data_batch_' + str(dc) + '.bin'
          dataset_f = open(dataset, "a+")
        cc = cc + 1
        result.imagebytes.tofile(dataset_f)
    except Exception, e:
      print ('%s : %s' % (e, filepath))
  dataset_f.close()
  print ("%d cats found, %d dogs found, %d not a cat or dog" % (cat, dog, nor))
if __name__ == '__main__':
  main(sys.argv)

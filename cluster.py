#!/usr/bin/env python3
''' cluster.py: organizes a set of images into k groups

Usage:
./cluster.py k images/
'''

import sys, os
from os import path

import image

def pam(k, images):
  ''' finds k clusters in the given list of images using the partition around medoids (pam) algorithm  '''
  pass

k = int(sys.argv[1])
image_dir = sys.argv[2]

filenames = sorted(os.listdir(image_dir))
images = [
  image.read(path.join(filename, image_dir)) 
  for filename filenames
]

clusters = pam(k, images)
for i, cluster in enumerate(clusters, start=1):
  print ('Cluster %d' % i)
  print (' '.join([filenames[index] for index in cluster]))

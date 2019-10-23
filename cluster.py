#!/usr/bin/env python3
''' cluster.py: organizes a set of images into k groups

Usage:
./cluster.py k images/
'''

import sys, os
from os import path
from operator import itemgetter
from tqdm import tqdm, trange

import numpy as np

import image

def pam(k, D, iter_count=50):
  ''' finds k clusters in the given list of images using the partition around medoids (pam) algorithm  '''
  n = len(D)

  indices = list(range(n))
  centers = np.random.choice(indices, size=k, replace=False)
  
  assignments = [d[centers].argmin() for d in D]
  error = sum(D[i,c] for i, c in enumerate(assignments))

  for _ in range(iter_count):
    for cid in range(k):
      candidates = list(set(indices) - set(centers))
      errors = []
      old_centers = list(centers)
      for candidate in candidates:
        centers[cid] = candidate
        assignments = [d[centers].argmin() for d in D]
        errors.append(sum(D[i,c] for i, c in enumerate(assignments)))

      min_index, min_error = min(enumerate(errors), key=itemgetter(1))
      if min_error < error:
        centers[cid] = candidates[min_index]
      else:
        centers = old_centers

      assignments = [d[centers].argmin() for d in D]

  clusters = [[i for i, c in enumerate(assignments) if c == cid] for cid in range(k)]
  return clusters

k = int(sys.argv[1])
image_dir = sys.argv[2]

filenames = sorted(os.listdir(image_dir))
images = [
  image.read(path.join(image_dir, filename)) 
  for filename in filenames
]

gray = [
  image.to_grayscale(img)
  for img in images
]

results = [
  image.orb_features(img)
  for img in gray
]

keypoints, features = zip(*results)

n = len(features)
D = np.zeros((n, n))

for i in trange(n):
  for j in trange(i+1):
    dij = image.distance(features[i], features[j])
    dji = image.distance(features[j], features[i])
    D[i,j] = D[j,i] = dij+dji

# for d in D:
#  print (' '.join(filenames[i] for i in d.argsort()[:3]))


clusters = pam(k, D, iter_count=100)
for i, cluster in enumerate(clusters, start=1):
  print ('Cluster %d' % i)
  print (' '.join([filenames[j] for j in cluster]))


from operator import itemgetter
from heapq import nsmallest

import cv2
import numpy as np 

orb = cv2.ORB_create()

def read(filepath):
  ''' reads an image from given filepath and returns a np.array of pixel values '''
  return cv2.imread(filepath)

def write(filepath, img):
  ''' writes an np.array of pixel values an an image to given filepath '''
  cv2.imwrite(filepath, img)

def to_grayscale(img):
  ''' converts an rgb image to grayscale '''
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def orb_features(img, length=300):
  ''' returns the orb features of a grayscale image '''
  points, features = orb.detectAndCompute(img, None)
  # point_feature = sorted(zip(points, features), key=lambda x: x[0].response, reverse=True)
  # points, features = zip(*point_feature)
  return points[:length], features[:length]


def match(features1, features2, threshold=0.75):
  ''' matches orb features and returns a list of matches  '''
  
  matches = []
  for i, f1 in enumerate(features1):
    norms = [cv2.norm(f1, f2, cv2.NORM_HAMMING) + 1e-6 for f2 in features2]
    distances = nsmallest(2, norms)
    
    normalized = distances[0]/distances[1]
    
    if normalized < threshold:
      matches.append((i, np.argmin(norms), normalized))

  return matches

def distance(features1, features2):
  ''' computes the non-symmetric distance between two images in orb feature space '''
  matches = match(features1, features2)
  match_count = len(matches)
  point_count = len(features1)

  return 1 - match_count/point_count

  

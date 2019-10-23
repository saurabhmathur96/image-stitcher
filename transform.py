import numpy as np
from itertools import product

def bilinear_interpolation(img, x, y):
  l, m = int(x), int(y)
  a, b = x-l, y-m

  try:
    return (1-b)*(1-a)*img[l,m] + (1-b)*a*img[l+1,m] +\
          b*(1-a)*img[l,m+1] + b*a*img[l+1,m+1]
  except IndexError:
      return [0, 0, 0] if len(img.shape) == 3 else 0

def apply(img, T):
  ''' applies a transform matrix T to a given image '''
  
  h, w, *_ = img.shape
  result = np.zeros((h, w), dtype=np.int8)
  J = np.array([(x,y,1) for x,y in product(range(h), range(w))]).T

  Tinv = np.linalg.pinv(T)
  I = np.dot(Tinv, J)
  I = I[:-1] / I[-1]

  for (xi, yi), (xj, yj) in zip(I.T, J[:-1].T):
    if not (0 <= xi < w or 0 <= yi < h): continue

    result[int(round(xj)), int(round(yj))] = bilinear_interpolation(img, xi, yi)

  return result




def translation(img1, img2, points):
  ''' finds a transformation matrix for len(points) = 1 '''
  pass

def rigid(img1, img2, points):
  ''' finds a transformation matrix for len(points) = 2 '''
  pass

def affine(img1, img2, points):
  ''' finds a transformation matrix for len(points) = 3 '''
  pass

def projective(img1, img2, points):
  ''' finds a transformation matrix for len(points) = 4 '''
  pass


def ransac(img1, img2):
  ''' computes the transformation matrix from img1 to img2 using ransac algorithm '''
  pass

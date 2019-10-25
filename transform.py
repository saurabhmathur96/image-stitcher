import numpy as np
from itertools import product
from operator import itemgetter
import image
from tqdm import trange

def change_axis_system(M):
  N = np.array([M[1], M[0], M[2]]).T 
  return np.array([N[1], N[0], N[2]]).T

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
  T = change_axis_system(T)
  if len(img.shape) == 2:
    h, w = img.shape
    result = np.zeros_like(img)
  else:
    h, w, c = img.shape
    result = np.zeros_like(img)
  J = np.array([(x,y,1) for x,y in product(range(h), range(w))]).T

  Tinv = np.linalg.pinv(T)
  I = np.dot(Tinv, J)
  I = I[:-1] / I[-1]

  for (yi, xi), (yj, xj) in zip(I.T, J[:-1].T):
    if ((0 <= round(xi) < w) and (0 <= round(yi) < h)):
      result[yj, xj] = bilinear_interpolation(img, yi, xi)

  return result




def translation(points1, points2):
  ''' finds a transformation matrix for len(points) = 1 '''
  [(x1, y1)] = points1
  [(x2, y2)] = points2
  T = np.eye(3)
  T[0, 2] = p2[0] - p1[0] # tx
  T[1, 2] = p2[1] - p1[1] # ty
  return T

def rigid(points1, points2):
  ''' finds a transformation matrix for len(points) = 2 '''
  [(x11, y11), (x12, y12)] = points1
  [(x21, y21), (x22, y22)] = points2
  A = np.array([
    [x11, -y11, 1, 0],
    [x12, -y12, 1, 0],
    [y11,  x11, 0, 0],
    [y12,  x12, 1, 0]
  ])

  B = np.array[[x21, x22, y21, y22]].T
  X = np.linalg.solve(A, B)
  x = X.reshape(-1)

  T = np.eye(3)
  T[0] = np.array([x[0], -x[1], x[2]])
  T[1] = np.array([x[1], x[0], x[3]])
  return T

def affine(points1, points2):
  ''' finds a transformation matrix for len(points) = 3 '''
  [(x11, y11), (x12, y12), (x13, y13)] = points1
  [(x21, y21), (x22, y22), (x23, y23)] = points2

  A = np.array([
    [x11, y11, 1, 0, 0, 0],
    [x12, y12, 1, 0, 0, 0],
    [x13, y13, 1, 0, 0, 0]
    [0, 0, 0, x11, y11, 1],
    [0, 0, 0, x12, y12, 1],
    [0, 0, 0, x13, y13, 1]
  ])
  B = np.array([[x21, x22, x23, y21, y22, y23]]).T
  X = np.linalg.solve(A, B)
  x = X.reshape(-1)

  T = np.eye(3)
  T[0] = np.array([x[0], x[1], x[2]])
  T[1] = np.array([x[3], x[4], x[5]])

  return T

def projective(points1, points2):
  ''' finds a transformation matrix for len(points) = 4 '''
  [(x11, y11), (x12, y12), (x13, y13), (x14, y14)] = points1
  [(x21, y21), (x22, y22), (x23, y23), (x24, y24)] = points2

  A = np.array([
    [x11, y11, 1, 0, 0, 0, -x11*x21, -y11*x21],
    [x12, y12, 1, 0, 0, 0, -x12*x22, -y12*x22],
    [x13, y13, 1, 0, 0, 0, -x13*x23, -y13*x23],
    [x14, y14, 1, 0, 0, 0, -x14*x24, -y14*x24],
    [0, 0, 0, x11, y11, 1, -x11*y21, -y11*y21],
    [0, 0, 0, x12, y12, 1, -x12*y22, -y12*y22],
    [0, 0, 0, x13, y13, 1, -x13*y23, -y13*y23],
    [0, 0, 0, x14, y14, 1, -x14*y24, -y14*y24],

  ])
  B = np.array([[x21, x22, x23, x24, y21, y22, y23, y24]]).T
  X = np.linalg.solve(A, B)
  x = X.reshape(-1)

  T = np.eye(3)
  T[0] = np.array([x[0], x[1], x[2]])
  T[1] = np.array([x[3], x[4], x[5]])
  T[2] = np.array([x[6], x[7], 1])

  return T


def compute_agreement(T, points1, points2, threshold=10):
  
  points1 = np.array([[point[0], point[1], 1] for point in points1]).T
  points2 = np.array([[point[0], point[1], 1] for point in points2]).T
  
  T = change_axis_system(T)
  transformed1 = np.dot(T, points1)
  transformed1 = transformed1[:-1] / transformed1[-1]

  #print (points2[:-1].shape)
  #print (transformed1.shape)
  distance = np.array([np.linalg.norm(p1-p2) for p1, p2 in zip(transformed1.T, points2[:-1].T) ])
  return sum(distance < threshold)

def ransac(points1, features1, points2, features2, round_count=1000):
  ''' computes the transformation matrix from img1 to img2 using ransac algorithm '''
  matches = image.match(features1, features2)
  match_count = len(matches)
  print (match_count)
  hypotheses = []
  points1 = np.array([[p.pt[0], p.pt[1] ] for p in points1])
  points2 = np.array([[p.pt[0], p.pt[1] ] for p in points2])
  for r in trange(round_count):
    try:
      indices = np.random.choice(range(match_count), 4, replace=False)
      print (indices)
      sample_matches = [matches[i] for i in indices]
      sample_points1, sample_points2 = zip(*[[points1[first], points2[second]] for first, second, _ in sample_matches])

      T = projective(sample_points1, sample_points2)
      votes = compute_agreement(T, points1, points2)
      
    except np.linalg.LinAlgError:
      pass

    else:
      hypotheses.append((T, sample_points1, sample_points2, votes))
  return max(hypotheses, key=itemgetter(3))
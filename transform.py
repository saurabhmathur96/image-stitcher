def apply(img, T):
  ''' applies a transform matrix T to a given image '''
  pass

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

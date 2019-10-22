def read(filepath):
  ''' reads an image from given filepath and returns a np.array of pixel values '''
  pass

def write(filepath):
  ''' writes an np.array of pixel values an an image to given filepath '''
  pass

def to_grayscale(img):
  ''' converts an rgb image to grayscale '''
  pass

def orb_features(img):
  ''' returns the orb features of a grayscale image '''
  pass

def match(features1, features2, threshold):
  ''' matches orb features and returns a list of matches  '''
  pass
  
def distance(features1, features2):
  ''' computes the distance between two images in orb feature space '''
  pass

import matplotlib.pyplot as plt
import numpy as np
import cv2

import transform, image


img1 = image.read('book1.jpg')
img2 = image.read('book2.jpg')


points1, features1 = image.orb_features(image.to_grayscale(img1), length=1000)
points2, features2 = image.orb_features(image.to_grayscale(img2), length=1000)

'''
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.scatter([point.pt[0] for point in points1], [point.pt[1] for point in points1], c='r', s=40)

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.scatter([point.pt[0] for point in points2], [point.pt[1] for point in points2], c='r', s=40)

plt.show()
'''

#T, points1, points2, votes = transform.ransac(points1, features1, points2, features2)
#T = transform.change_axis_system(T)
matches = image.match(features1, features2)

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.scatter([points1[m[0]].pt[0] for m in matches], [points1[m[0]].pt[1] for m in matches], s=40)

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.scatter([points2[m[0]].pt[0] for m in matches], [points2[m[1]].pt[1] for m in matches],  s=40)

plt.show()

print (T, votes)
img = image.to_grayscale(img1)

'''
points = '318,256 141,131 534,372 480,159 316,670 493,630 73,473 64,601'
points = [list(map(int, each.split(","))) for each in points.split()]

T = transform.projective(points[0::2], points[1::2])
T = transform.change_axis_system(T)
print (transform.compute_agreement(T, points[0::2], points[1::2]))
#
'''




plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
transformed = transform.apply(img, T)
plt.imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

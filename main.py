import numpy as np
import imageio.v3 as iio # to read and write images
import matplotlib.pyplot as plt


# Other files in the project
from harris_corner_detector import detect_features
from NCC import find_correspondances
from homography_RANSAC import estimate_homography
from merge_images import warp_image1_onto_image2



def rgb_to_gray(rgb):

    # compute a weighted average of RGB colors to obtain a greyscale value
    # weights correspond to the luminosity of each color channel
    # we also normalize the image
    return (1/255)*np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

##################
# Read the two images
##################

image1 = iio.imread(uri='DanaHallWay1/DSC_0281.JPG')
image2 = iio.imread(uri='DanaHallWay1/DSC_0282.JPG')

grey_img1 = rgb_to_gray(image1)
grey_img2 = rgb_to_gray(image2)

#TODO consider reducing the size here

##################
# Detect features of both images
##################

# get masks with corner locations
features1 = detect_features(grey_img1)
features2 = detect_features(grey_img2)

# visualize the detected corners
plt.imshow(image1, cmap='gray')
plt.plot(features1[:, 1], features1[:, 0], 'r.', markersize=5)
plt.show()

plt.imshow(image2, cmap='gray')
plt.plot(features2[:, 1], features2[:, 0], 'r.', markersize=5)
plt.show()

##################
# Find correspondaces between the two sets of cornes
##################

# get a list containing all the feature correspondances
corresp = find_correspondances(grey_img1, grey_img2, features1, features2)

print(corresp)


"""
DRAW THE CORRESPONDENCES BETWEEN TWO IMAGES

from skimage import draw

img3 = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1]))
img3[:img1.shape[0], :img1.shape[1]] = img1
img3[:img2.shape[0], img1.shape[1]:] = img2

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(img3, cmap='gray')

for match in good_matches:
    idx1 = match[0]
    idx2 = match[1] + img1.shape[1]
    r, c = draw.line(kp1[idx1][0], kp1[idx1][1], kp2[idx2-img1.shape[1]][0], kp2[idx2-img1.shape[1]][1])
    ax.plot(c, r, linewidth=0.4, color='blue')

plt.show()
"""
##################
# Estimate the homography using RANSAC to ignore outliers
##################

# estimate the homography matrix from the computed correspondances
homography = estimate_homography(corresp)

##################
# Warp one image onto the other to obtain the final mosaic
##################

# obtain the final mosaic with the computed homography
mosaic = warp_image1_onto_image2(image1, image2, homography)

# write the mosaic to disk
# iio.imwrite(uri="output/mosaic.png", image=mosaic)


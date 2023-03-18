import numpy as np
import imageio.v3 as iio # to read and write images
import matplotlib.pyplot as plt
from skimage import draw
import cv2

# Other files in the project
from algorithm_homography import homography
from harris_corner_detector import detect_features
from merge_images import warp_images


def rgb_to_gray(rgb):

    # compute a weighted average of RGB colors to obtain a greyscale value
    # weights correspond to the luminosity of each color channel
    # we also normalize the image
    return (1/255)*np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


##################
# Read the three images
##################

images = [iio.imread(uri='DanaOffice/DSC_0308.JPG'),
          iio.imread(uri='DanaOffice/DSC_0309.JPG'),
          iio.imread(uri='DanaOffice/DSC_0310.JPG')]

# convert to grey images and reduce them by a scaling factor
grey_images = []
scale = 0.65

for i in range(len(images)):
    h, w = images[i].shape[0], images[i].shape[1]
    images[i] = cv2.resize(images[i], (int(scale*w), int(scale*h)))
    grey_images.append(rgb_to_gray(images[i]))


##################
# Detect features of each image
##################

features = []

for i in range(len(images)):
    features.append(detect_features(grey_images[i]))

    # visualize the detected corners
    plt.imshow(images[i], cmap='gray')
    plt.plot(features[-1][:, 1], features[-1][:, 0], 'r.', markersize=5)
    plt.show()


# compute the homographies to do the merging
# we will warp img1 and img3 into img2

# homography1: img1 -> img2
homography1 = homography(grey_images[0], grey_images[1],
                        features[0], features[1])
# homography1: img3 -> img2
homography3 = homography(grey_images[2], grey_images[1],
                        features[2], features[1])

##################
# Warp one image 1 and 3 onto image 2 the final mosaic
##################

# obtain the final mosaic with the computed homography
mosaic = warp_images(images[0], images[1], images[2], homography1, homography3)

# Display the output mosaic
plt.imshow(mosaic)
plt.show()

# write the mosaic to disk
iio.imwrite(uri="output/mosaic.png", image=mosaic)

